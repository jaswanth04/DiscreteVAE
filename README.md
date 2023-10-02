## Discrete VAE

For more information on DiscreteVAE, please visit this medium article. (Link to be updated)

## Training



### Dataset used for training
We used the following dataset for training - https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set

### Configurations
Deepspeed has been used to help in training of this model. The deepspeed configurations is present in [deepspeed_configs folder](https://github.com/jaswanth04/DiscreteVAE/tree/master/deepspeed_configs)

### Start the training procedure

To start the training procedure use the below code

```
deepspeed train.py 
```

## Inferencing

Please look at the [notebook](https://github.com/jaswanth04/DiscreteVAE/blob/master/notebooks/inference.ipynb)

For any questions or feedback, please raise an Issue or mail me at jaswanth04@gmail.com