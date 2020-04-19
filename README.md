### This repository has no license, therefore all rights are reserved. You cannot modily or redistribute this code without explicit permission from the copyright holder.
### Please send an e-mail to: `realtimeactionrecognition@gmail.com` for any queries.


# Real-time Action Recognition System
This is a specific use-case of the Real-time Action Recognition System. In this implementation, we use the new model to demostrate handwash auditing, by using a live feed of a camera setup above a sink where the handwash is being performed.

## Instructions to run the code:

#### 1. Install the requirements:
a. In the parent directory, run:
    
      $ pip install -r requirements.txt
      
      
      
#### 2. Compile and setup PyFlow:
a. Change directory to `pyflow`

      $ cd pyflow
      
b. Setup `pyflow`:

      $ python setup.py build_ext -i
    
    
    
#### 3. Dowload the trained model
a. Download the model from [here](https://drive.google.com/file/d/1PdOtBNrUbt4t9o3RPZm1kCZ8KRoziFnV/view?usp=sharing).

b. Save the model in the parent folder of this repository, with the name: `current_final_handwash_model.h5`



#### 4. Run the Flask app:

      $ python app.py
      
      
      
#### 5. Open `localhost:8001` on your browser when the `* Debugger pin is active` message appears on the Terminal.


## Training and Testing Data:

The training and testing data used was the first split of the UCF-101 Dataset. Due to computation limitations, the first split of the UCF-101 dataset has been preprocessed accoring to our implementation, and stored in the following locations as `.pyc` files:
	
The pre-processed training data can be found [here](https://drive.google.com/file/d/1Xh5sL50sR8qNT5V9lQLqN683pz3JkJg_/view?usp=sharing).

The pre-processed testing data can be found [here](https://drive.google.com/file/d/1z11xSq9VZlPKKj33eTAfGyzp82B1pJjY/view?usp=sharing).



## Points to remember:
1. A webcam must be connected to the computer/laptop.
2. The model must be saved in this repository's parent folder.
3. The computer must have internet access, since JavaScript is dynamically linked.


