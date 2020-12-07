# VAE, GAN, Domain Adaptation pytorch 

**Implement**
* Variance AutoEncoder
* Generative Adversarial Network
* Domain Adversarial Neural Network 
* Domain Seperation Network
* Adversarial Discriminative Domain Adaptation

# Get Dataset
```bash
bash get_dataset.sh
```

# Argparse Param 
* --resume : resume model path 
* --b : batch size
* --j : number of workers
* --epoch : number of epochs
* --model : model name
* --task : which task
* --src : source domain name : [usps, mnistm, svhn] ⚠️ ***only for DANN task*** ⚠️ 
* --tar : target domain name : [usps, mnistm, svhn] ⚠️ ***only for DANN task*** ⚠️ 
* --ep_src : number of pretrained epoch ⚠️ ***only for ADDA model*** ⚠️ 

# Train 
###### Variance AutoEncoder(VAE)
```python
python main.py --task VAE --model VAE 
```
###### Generative Adversarial Network(GAN)
```python
python main.py --task GAN --model GAN 
```
###### Domain Adversarial Neural Network 
```python
python main.py --task DANN --model DANN --src [source name] --tar [target name]
```
###### Domain Seperation Network
```python
python main.py --task DANN --model DSN --src [source name] --tar [target name]
```
###### Adversarial Discriminative Domain Adaptation
```python
python main.py --task DANN --model ADDA --src [source name] --tar [target name]
```

# Reference 
Below is original paper link :
> [`Variance Autoencoder`](https://arxiv.org/abs/1312.6114)
> [`Generative Adversarial Network`](https://arxiv.org/abs/1406.2661)
> [`Domain Adversarial Neural Network`](https://arxiv.org/abs/1505.07818`](https://arxiv.org/abs/1505.07818)
> [`Domain Seperation Network`](https://papers.nips.cc/paper/2016/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf`](https://papers.nips.cc/paper/2016/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf)
> [`Adversarial Discriminative Domain Adaptation`](https://arxiv.org/abs/1702.05464`](https://arxiv.org/abs/1702.05464)

