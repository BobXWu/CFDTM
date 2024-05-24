## **Check our latest topic modeling toolkit [TopMost](https://github.com/bobxwu/topmost) !**


# Code for Modeling Dynamic Topics in Chain-Free Fashion by Evolution-Tracking Contrastive Learning and Unassociated Word Exclusion (ACL 2024 Findings)

## 1. Prepare environment

    python==3.8.0
    pytorch==1.7.1
    scikit-learn==1.0.2
    gensim==4.3.0
    pyyaml==6.0
    tqdm

Notice: Fix the bug of invalid values of coherence models in gensim following [RaRe-Technologies/gensim#3040](https://github.com/RaRe-Technologies/gensim/issues/3040#issuecomment-812913521).


## 2. Train and evaluate the model

We provide a shell script under `./CFDTM/scripts/run.sh` to train and evaluate our model.  
Change to directory `./CFDTM`, and run command as

    ./scripts/run.sh NYT 50


Other datasets are available in [TopMost](https://github.com/BobXWu/TopMost/tree/main/data).



## Citation

If you want to use our code, please cite as

    @article{wu2024dynamic,
        title={Modeling Dynamic Topics in Chain-Free Fashion by Evolution-Tracking Contrastive Learning and Unassociated Word Exclusion},
        author={Wu, Xiaobao and Dong, Xinshuai and Pan, Liangming and Nguyen, Thong and Luu, Anh Tuan},
        journal={arXiv preprint},
        year={2024}
    }
