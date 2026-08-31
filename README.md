# ML-BE-fitness-score
Machine Learning-Based Prediction of Base Editor sgRNA fitness score

Source code for reproduce results of


Machine Learning-Based Prediction of Base Editor sgRNA fitness score

Alessandro Orro, Arianna Consiglio, Maria Ilaria Curci, Martina Scichilone, Faiza Hasin, Michele Minervini, Corrado Mencar, Gianluca De Bellis, Cinzia Cocola, Paride Pelucchi, Tommaso Selmi

## Requirements
    python==3.13

    biopython==1.85
    catboost==1.2.8
    h5py==3.13.0
    pandas==2.3.3
    joblib==1.4.2
    lightgbm==4.6.0
    numpy==1.26.4
    scikit-base==1.0.2
    scikit-learn==1.5.2
    scipy==1.17.1
    tqdm==4.67.1
    statsmodels==0.14.6
    shap==0.47.2
    matplotlib==3.10.7
    xgboost==3.0.0
    openpyxl==3.1.5

## Dataset format
The data input is a x.sx file with the following columns

*Gene symbol*
Symbol identifying the target gene associated with the genomic locus or sgRNA target.
    
    sgRNA sequence: Nucleotide sequence of the single guide RNA (typically 20 nucleotides in length).
    
    sgRNA strand: Genomic strand orientation of the sgRNA (sense or antisense).
    
    Gene strand: Orientation of the target gene transcript (1 for sense, -1 for antisense or similar numerical encoding).
    
    Mutation bin: Functional classification or predicted variant effect category of the target site (e.g., Intron, Missense, No edits, Nonsense, Silent, Splice site, UTR).
    
    Nucleotide edits: List or representation of specific nucleotide changes introduced or evaluated at the target site.
    
    On-target efficacy score: Benchmark cleavage or on-target score from Rule Set 2.
    
    efficiency:behive:BE4: Predicted base editing efficiency score from the BE-Hive model for BE4.
    
    efficiency:bedict:BE4max: Predicted base editing efficiency score from the BE-DICT model for BE4max.
    
    efficiency:deepbe:NG: Predicted base editing efficiency score from the DeepBE model for NG variants.
    
    domains:pfam: Protein domain annotation identifier or description from the PFAM database.
    
    DepMap:[Cell_Line]: Cancer Dependency Map (DepMap) score indicating gene essentiality or cell fitness effect for specific cell lines (e.g., A375, MELJUSO, OVCAR8, HAP1, HA1E).
    
    BE39:[Cell_Line]:zscore: Experimental proliferation or depletion z-score measured under BE3.9 base editing conditions for a given cell line.
    
    CAS9:[Cell_Line]:zscore: Experimental proliferation or depletion z-score measured under standard Cas9 nuclease conditions for a given cell line.
    
    context[N]: Flanking genomic sequence context window of length N (e.g., 20, 30, 40, 50 bp) centered around the target site.

 
