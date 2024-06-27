# Real-time Robust State Estimation for Large-scale Low-observability Power-transportation System Based on Meta Physics-informed Graph TimesNet

## Abstract
> *The ever-increasing penetration of Electric Vehicles (EVs) has emerged as an evident challenge within the Power Distribution System (PDS). Existing Distribution System State Estimation (DSSE) models are constrained by the low-observability and often neglect the spatio-temporal correlations inherent in the PDS. To address this, our study proposes a robust learning architecture named meta physics-informed graph TimesNet that can effectively learn the evolving topology of the PDS, as well as capture the spatio-temporal correlations of the PDS state, even under the condition of limited data availability. The model leverages a combination of a graph convolutional network and TimesNet as the foundational modules for constructing an encoder-decoder framework. Furthermore, to capture the spatio-temporal heterogeneity of the PDS state, our study inserts a topology learning module driven by a meta physics-informed graph bank into the encoder-decoder framework. Subsequently, power flow calculations are performed on the predicted power to reduce error accumulation. The IEEE 8500-node test feeder and UTD19 are utilized as the datasets for the PDS and urban transportation system, respectively. By using 42 electric vehicle charging stations as coupling points, a novel large-scale power-transportation coupled system dataset is constructed through multi-source information fusion. The performance of our proposed model on this dataset has shown to surpass existing DSSE methods in terms of accuracy (MAPE, etc.) and effectiveness, particularly considering the increasing prevalence of EVs.*


<img width="1098" alt="image" src="https://github.com/lishijie15/MPGTN-for-DSSE/pictures/MPGTN.pdf">



#### Requirements

Model is compatible with PyTorch==1.13 versions.

Please code as below to install some nesessary libraries.

```
pip install -r requirements.txt
```

#### Run Our Model

Firstly, perform the prediction component. Ensure you are in the `./MPGTN` directory before executing the following code.

```
python MPGTN_main.py
```

To execute the power flow calculation, run the following code and ensure you are in the `./` directory. (Note: Power flow calculation can only be performed on Windows.)

```
python test_final_ture&pre.py
```

Control the load flow parameters by modifying the Master.dss file.
