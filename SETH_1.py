#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
import torch
import transformers
from transformers import T5EncoderModel, T5Tokenizer
import requests
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
print("IMPORTANT: this will be EXTREMELY slow if not run on GPU. (so the above should say sth like cuda:0).")

#Define CNN structure
class CNN( nn.Module ):
    def __init__( self, n_classes, n_features, pretrained_model=None ):
        super(CNN, self).__init__()
        self.n_classes = n_classes
        bottleneck_dim = 28       
        self.classifier = nn.Sequential(
                        #summarize information from 5 neighbouring amino acids (AAs) 
                        #padding: dimension corresponding to AA number does not change
                        nn.Conv2d( n_features, bottleneck_dim, kernel_size=(5,1), padding=(2,0) ), 
                        nn.Tanh(),
                        nn.Conv2d( bottleneck_dim, self.n_classes, kernel_size=(5,1), padding=(2,0))
                        )

    def forward( self, x):
        '''
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of output nodes (1 for disorder, since predict one continuous number)
        '''
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1) 
        Yhat = self.classifier(x) # OUT: Yhat_consurf = (B x N x L x 1)
        # IN: (B x N x L x 1); OUT: ( B x L x N )
        Yhat = Yhat.squeeze(dim=-1)
        return Yhat


def read_fasta(fasta_path):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.strip()
                sequences[uniprot_id] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[uniprot_id] += ''.join(line.strip().split()).upper()

    return sequences


def get_prott5(root_dir):
    print("Loading ProtT5...")
    transformers.logging.set_verbosity_error()
    #excluded lines are alternative import routes
    #cache_dir = root_dir / "ProtT5_XL_U50"
    #cache_dir.mkdir(exist_ok=True)
    transformer_link = "Rostlab/prot_t5_xl_uniref50"
    #model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=cache_dir) 
    model = T5EncoderModel.from_pretrained(transformer_link)
    if not device.type=='cpu':
        model = model.half() # run in half-precision to reduce vRAM consumption
    model = model.to(device)
    model = model.eval() # run in evaluation mode to ensure determinism
    #tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, cache_dir=cache_dir)
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    return model, tokenizer


def load_CNN_ckeckpoint(root_dir):
    print("Loading SETH_1...")
    predictor=CNN(1, 1024)
    checkpoint_dir = root_dir / "CNN"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_p = checkpoint_dir / 'CNN.pt'
    if not checkpoint_p.is_file():
        url="https://rostlab.org/~deepppi/SETH_CNN.pt"
        with requests.get(url, stream=True) as response, open(checkpoint_p, 'wb') as outfile:
            outfile.write(response.content)
    state = torch.load( checkpoint_p )
    predictor.load_state_dict(state['state_dict'])
    if not device.type=='cpu':
        predictor = predictor.half() # run in half-precision
    predictor = predictor.to(device)
    predictor = predictor.eval()
    return predictor
    

def get_predictions(seqs, prott5, tokenizer, CNN):
    print("Making predictions...")
    
    predictions = dict()
    for protein_id, original_seq in seqs.items():
        seq = original_seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))

        token_encoding = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding="longest")
        input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        try:
            with torch.no_grad():
                # get embeddings extracted from last hidden state 
                emb = prott5(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,:seq_len] # [1, L, 1024]
                # predict Z-scores with CNN
                prediction=CNN(emb).detach().squeeze().cpu().numpy()
                # convert Z-scores into 0,1 with threshold 8, disorder=1
                diso_pred=(prediction<8).astype(int)
                # confidence metric: Z-scores normalized to [0,1]
                prediction=prediction*(-1) # disorder should have higher numbers than order
                confidence=(prediction-(-17))/(6-(-17))
                #all confidence values smaller than 0, larger than 1 mapped to 0 or 1.
                for i in range(len(confidence)):
                    if confidence[i]<0: 
                        confidence[i]=0
                    if confidence[i]>1:
                        confidence[i]=1
                predictions[protein_id] = (original_seq, diso_pred, confidence)
        except RuntimeError as e :
            print(e)
            print("RuntimeError during embedding for {} (L={})".format(protein_id, seq_len))
            continue
    return predictions


def write_predictions(out_path, predictions):
    with open(out_path, 'w+') as out_f:
        for protein_id, (sequence, prediction, confidence) in predictions.items():
            out_string = [protein_id]
            for idx, AA in enumerate(sequence):
                binary = prediction[idx]
                conf=confidence[idx]
                out_string.append("{}\t{}\t{:.4f}\t{}".format(
                    idx+1, AA, conf, round(binary)))
            out_f.write( "\n".join(out_string) + "\n")
    return None


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
        'caid_diso.py classifies residues in a given protein sequence into ' +
        'order[0] or disorder[1] for the CAID2 (2022) challenge .'))

    # Required positional argument
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='A path for saving the disorder predictions as CSV in CAID format.')
    return parser


def main():
    root_dir = Path.cwd()
    
    parser = create_arg_parser()
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    seqs = read_fasta(in_path)
    prott5, tokenizer = get_prott5(root_dir)
    CNN = load_CNN_ckeckpoint(root_dir)
    predictions = get_predictions(seqs, prott5, tokenizer, CNN)
    write_predictions(out_path, predictions)


if __name__ == '__main__':
    main()
