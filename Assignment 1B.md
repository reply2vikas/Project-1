Q1- What are Channels and Kernels (according to EVA)?
Ans - 
  Channels -
    If there are similar kind of information then we call it cannels.
    Ex; 1. In the dish of Pulou, Pee, Rice, Onion are the channels as a ingredients.
        2. Different music instruments but purpose is same.
        3. Different TV channels.
  
  Kernels - 
    
    Kernels are the brains in a convolutional neural network. They can range from any shapes like 1x1, 3x3 etc. depending on the image. A     kernel based on its soze looks at a matching size in the larger image. Tt extracts information from the part that it has observed on       and saves them, These are the weights for the input. To squish down the features learnt to make meaningful decisions we use a process     called MaxPooling where in we take a value which is the maximum of the values it has in the filter. Using a Maxpool layer decreases       the dimentionality of the image and kernels of different sizes can be further applied on it. Filter parametes inclues the size of the     filter, the type of activation it uses, the stride upon which it scans. A Cnn can have any number of layers depending upon the compute     power of the workstation, As the number of layers increases, the filters can lean more and more features from the image and yield         better results.
    As per the EVA, the example of Kernals are-
    1. No of Pee
    2. Amount of rice.
    3. Onion, etc.
Q2 - Why should we only (well mostly) use 3x3 Kernels?
     To get more optimization we must use 3x3 kernels. It runs super fast on networks. Nvidia and Intel are using it on their projects.

Q3 - How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations).
     400 | 398 | 396 | 394 | 392 | 390 | ... MP(2x2)
     195 | 193 | 191 | 189 | 187 | 185 | ... MP(2x2)
     ...
     .
     .
     .
     .
      ;-)
     
     
