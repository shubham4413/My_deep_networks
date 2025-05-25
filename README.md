# My_deep_networks
Here, I upload the classification and regression tasks I have worked on using deep neural networks.

## LeNet

This is the first Convolutional Neural Network I implemented using Pytorch. This is in the Jupyter Notebook titled MNIST Classification.ipynb. This familiarized me with the basic structure of coding neural networks with Pytorch. I learned how to load the data, create my custom Model, and add convolution layers, linear layers, and activation functions. This taught me how to write the learning algorithm to get the losses and plot them to evaluate the model and check if it is learning or overfitting and how to increase the model's accuracy. I also implemented a similar architecture using Jax, the code is available in jax_mnist.ipbyn python notebook. I learned how to use Jax and Flax libraries to create a deep neural network architecture and how to write a training loop with it and I added dropout in it to get a better result on test accuracy.

## Convolutional-Recurrent Neural Network

For this code, I have implemented a combination of a Convolutional Neural Network, Gated recurrent unit, and a fully connected layer to classify the paintings from the [WikiArt dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md) based on the artists. The code is under the jyputer notebook CRNN_Class_custom.ipynb. This architecture was chosen to extract the local features of the painting using the CNN and pass it through GRU to learn dependency within the painting and then, finally, a FC layer for classification. This task taught me how to pre-process the data and load it into a datloader to make the data ready for training. I also experimented with a lot of CNN architectures like pre-trained ResNets and VGGNets, but since they were trained on the Imagenet Dataset, which is very different from the abstract paintings of WikiArt, training a custom CNN is the better choice. To prevent the overfitting of data, I implemented dropout and early-stopping strategies. To ensure better learning, I also used learning rate scheduling so the learning can be slower if it is near a global minimum.  

## Similarity using BERT and MLP

This task aims to find similarities between paintings using [National Gallery of Art Open Data](https://github.com/NationalGalleryOfArt/opendata). The code is under the name Similarity.ipynb. The model learns to find similarities between structured data entries by encoding different types of features—textual, categorical, and numerical—into a shared embedding space. Textual data is processed through a pre-trained language model like BERT to capture semantic meaning, while categorical and numerical features are encoded using embeddings and a multi-layer perceptron (MLP), respectively. These representations are then fused (typically concatenated or averaged) into a single unified vector. During training, the model minimizes a similarity loss—like cosine similarity or contrastive loss—between pairs of related and unrelated entries. Over time, it learns to map semantically or structurally similar records closer together in the embedding space, enabling effective similarity comparisons.


## Fine-Tuning Llama 3.2 (1B) with Transformers & PEFT
This project involves fine-tuning the Llama 3.2 1B model using Hugging Face's transformers library and Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA (Low-Rank Adaptation). Key learnings include:

**PEFT Efficiency**: By freezing the base model and training only low-rank adapter layers, we achieved significant GPU memory savings (~70% less VRAM) while retaining performance.

**Optimized Training**: Using bitsandbytes for 4-bit quantization (QLoRA) further reduced hardware requirements, enabling fine-tuning on a single 16GB GPU.

**Challenges**: Balancing LoRA rank (r=8) and alpha (α=16) was critical to avoid underfitting/overfitting. Gradient checkpointing and gradient_accumulation_steps helped stabilize training.

## Setup
After cloning this repository and going to it's root, run this command:

`pip install -r requirements.txt`

This will install all the required libraries and you should be able to produce most of the code. A few things to keep in mind, this requires a minimum of `Python 3.9` and `CUDA 11.7`. To train `llama_lora_finetune.py ` file, the GPU should have a minimum of 16GB of RAM. 
