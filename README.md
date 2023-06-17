# FishNet-
Implementation of "FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction" [https://arxiv.org/abs/1901.03495] 

Task: Image Classification  
Dataset: CIFAR-10 [https://www.cs.toronto.edu/~kriz/cifar.html]  

to train, do python main.py  
--root_dir: root directory path where train, val, and test data folder exist  
main.py line 97-99: path where dataloader is saved  
train.py line 120, 123: path where best validation model is saved  

best validation achieved currently using fishnet150 
batch_size=64, optimizer AdamW lr 1e-4, augmentation random grayscale and random horizontal flip  
no pooling before first layer of fish tail (input res 32x32)  

Accuracy: 87.5%  
F1 weighted: 87.5%  
F1 per class:  
airplane	  88.7%  									
automobile	93.5%  									
bird				82.4%  						
cat					75.6%  			
deer				85.7%  				
dog					81.6%  			
frog				90.6%  				
horse				91.3%  				
ship				93.3%  				
truck       91.7%  


