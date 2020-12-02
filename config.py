config = {}

### ADDA
adda_config = {
	'src':1e-3,
	'tar':1e-4,
	'cls':1e-3,
	'dis':1e-3
}
config['ADDA'] = adda_config

### DANN
config['DANN'] = {'DANN' : 1e-3}

### DSN
config['DSN'] = {'DSN' : 1e-3}

### GAN
config['GAN'] = {
	'generator': 1e-3,
	'discriminator' : 1e-3
}

### VAE
config['VAE'] = {'VAE': 1e-3}
