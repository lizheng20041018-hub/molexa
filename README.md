# MOLEXA <br><br>
<img width="1811" height="1417" alt="Fig_1_annotated" src="https://github.com/user-attachments/assets/7f85b150-135e-45dc-ab69-08114cdc4096" />
<br><br>
MOLEXA is a deep generative neural network capable of reconstructing molecular geometries from ion momentum measurements in X-ray-based Coulomb explosion imaging (CEI) experiments. It is built on the Transformer architecture and the diffusion generative modeling framework. It allows for inversion of momentum-space datasets to position space, thus providing the structure of a molecule right before its interaction with an X-ray pulse. In addition, it can provide an uncertainty estimate for its reconstructed molecular geometries. By employing time-resolved CEI datasets, MOLEXA is able to provide ”snapshots” of a molecule at different instants during a chemical reaction. This enables the use of the CEI technique for direct reconstruction of molecular dynamics as they unfold on femtosecond or longer time scales.  
<br><br>
<br><br>

## Main architecture <br><br>
Dynamics Extraction Module  
<img width="1450" height="393" alt="image" src="https://github.com/user-attachments/assets/68f3eaf1-ada0-4c92-876d-dde236fa2aff" />     
<br><br>
Structure Denoising Module       
<img width="1509" height="505" alt="image" src="https://github.com/user-attachments/assets/23e3decf-9146-422f-93d8-fa67e47a52e1" />  
<br><br>
<br><br>
## Exemplary reconstructions <br><br>
The predicted and ground-truth structures are shown in opaque and semi-transparent colors, respectively.
<br><br>
<img width="1200" height="644" alt="image" src="https://github.com/user-attachments/assets/c03a8d5c-4beb-44e2-8a7f-0ab55fca8223" />
<br><br>
<br><br>
## Acknowledgement <br><br>
The training framework is based on Fastai's miniai library.







