# SETH
SETH is a predictor of nuances of a residue's disorder in proteins. As input, it only needs single protein sequences, which are then encoded in embeddings by the protein language model ProtT5 [[1]](#1) (Transformer). These embeddings are then passed to a two-layer CNN, whose output are the disorder predictions. The disorder predictions are given as CheZOD scores [[2]](#2) (or the normalization thereof, if you choose the default output format), which are continuous values for measuring disorder, where values below 8 signify disorder and values above 8 signify order [[2]](#2), [[3]](#3). If you choose the default output format, you will additionally receive binary disorder/order predictions. Since SETH only needs single protein sequences as input, it can be applied to any protein and predictions are very fast compared to most disorder predictors achieving high performances.  

## Usage
### Usage online
There is a [Google Colab](https://colab.research.google.com/drive/1vDWh5YI_BPxQg0ku6CxKtSXEJ25u2wSq?usp=sharing) script available, where you can execute SETH online to receive disorder predictions (CheZOD scores) and can subsequently download them.
### Usage on your local machine
If you have installed python and the libraries specified in the requirements.txt, the python script SETH_1.py can be executed in your cmd after download by typing `SETH_1.py -i <your input fasta file name> -o <the desired name of your output file>`.
You may optionally add `-f Cs` at the end, to only get CheZOD scores for the input protein(s) as output. By default, your output will be a table in [CAID](https://idpcentral.org/caid) format (per protein you get: one colum with the residue number, one colum with the amino acid name, one colum with normalized CheZOD scores and one colum with binary disorder/order predictions).

## Performance
When adding the performance of SETH to the large scale evaluation of disorder predictors [[3]](#3), i.e., evaluating SETH on the same test set, it outperformed all other methods.![](/images/Figure_3.jpg) 

## Cite
In case you are using this method and find it helpful, please cite... (will soon be available)

## References
<a id="1">[1]</a> 
Elnaggar, A., Heinzinger, M., Dallago, C., Rihawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik, D. & Rost, B. 2021. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing. arXiv:2007.06225

<a id="2">[2]</a> 
Nielsen, J. T. & Mulder, F. A. A. 2020. Quantitative Protein Disorder Assessment Using NMR Chemical Shifts. In: KRAGELUND, B. B. & SKRIVER, K. (eds.) Intrinsically Disordered Proteins. New York, NY: Springer US.

<a id="3">[3]</a> 
Nielsen, J. T. & Mulder, F. A. A. 2019. Quality and bias of protein disorder predictors. Scientific reports, 9, 5137