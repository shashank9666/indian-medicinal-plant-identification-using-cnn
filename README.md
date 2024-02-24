# Indian Medicinal Plant Identification using CNN
Project Description : This project aims to identify Indian medicinal plants using Convolutional Neural Networks (CNN). The project uses a Flask backend to serve the model and provide an interactive interface for users to upload images of plants for identification.  

 Dataset :The dataset used in this project consists of images of various Indian medicinal plants. You can find the dataset [here]  
https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset

# Installation and Setup
1. Clone the repository to your local machine.
   make a folder  called as 'static' and make 3 subfolders 'css','js','image' add files like javascript to js, add css files to css , image files to image, next create another folder named 'templates' in  same 
   directory where static was placed , add html files to it
3. Install the necessary dependencies using pip:
4. pip install flask , pip insatll tensorflow, pip install pillow ,etc
5. Run the Flask application:     ``` py app.py     ```

## Usage: After running the Flask application, navigate to the localhost URL in your web browser. Upload an image of the plant you want to identify, and the web application will return the predicted plant species along with it's medicinal uses
