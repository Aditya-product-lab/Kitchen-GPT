Welcome, this file will help you with all the steps needed to get your own toy model (Kitchen-GPT) up and running via the terminal interface on Microsoft Visual Studio Code (VS Code) or any other IDE. 
Here are the steps to follow: 

1. Make sure you have Python and Microsoft VS Code (or any other IDE) installed in your PC.

2. Download and paste the files "train.py" and "cooking.txt" to a new folder in your PC.

3. Open the folder through VS Code (or any other IDE) and create a virtual environment, Run the command: "python -m venv .venv". Then to activate it run the command: ".venv\Scripts\activate"

4. Install packages PyTroch. Run the command: "pip install torch" (mandatory) and later "pip install numpy" (optional but recommended) as these are needed for the code to run. 

5. Finally, type "python train.py" to initiate model training, once training is complete type a word or letter in the training model and the model will auto complete with its own few words. 

Note: Edit the hyperparameters on the code (line 8 to 12) and Temperature in line 268, retrain the model "Python train.py" to observe how it behaves under different settings. 
For any errors or to understand the code better use Claude Code, Gemini or Codex to help you debug. 
