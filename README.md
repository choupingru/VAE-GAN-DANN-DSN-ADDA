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
* --test : test mode (default 0)
* --task : which task
* --src : source domain name : [usps, mnistm, svhn] ⚠️ ***only for DANN task*** ⚠️ 
* --tar : target domain name : [usps, mnistm, svhn] ⚠️ ***only for DANN task*** ⚠️ 
* --ep_src : number of pretrained epoch ⚠️ ***only for ADDA model*** ⚠️ 


# Variance AutoEncoder(VAE)
```python
python main.py --task VAE --model VAE 
```

# Generative Adversarial Network(GAN)
```python
python main.py --task GAN --model GAN 
```

# Domain Adversarial Neural Network 
```python
python main.py --task DANN --model DANN 
```

# Domain Seperation Network
```python
python main.py --task DANN --model DSN
```

# Adversarial Discriminative Domain Adaptation
```python
python main.py --task DANN --model ADDA
```