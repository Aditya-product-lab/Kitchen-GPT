## What is Kitchen-GPT?

Think of **Kitchen-GPT** as an interactive, miniature AI kitchen laboratory. 

Large Language Models (like ChatGPT) feel like magic because their training happens behind closed doors across thousands of expensive supercomputers. **Kitchen-GPT demystifies this process.** It trains a real, functioning miniature Transformer model right in your browser on a tiny set of cooking sentences.

By tweaking simple sliders and training the model live, you can immediately chat with your newly created "chef" and see firsthand how small behind-the-scenes settings completely transform an AI's intelligence, memory, and creativity.

---

## How to get started?

Here are the steps to follow: 

1. Make sure you have Python and Microsoft VS Code (or any other IDE) installed in your PC.

2. Download and paste the files "train.py" and "cooking.txt" to a new folder in your PC.

3. Open the folder through VS Code (or any other IDE) and create a virtual environment, Run the command: "python -m venv .venv". Then to activate it run the command: ".venv\Scripts\activate"

4. Install packages PyTroch. Run the command: "pip install torch" (mandatory) and later "pip install numpy" (optional but recommended) as these are needed for the code to run. 

5. Finally, type "python train.py" to initiate model training, once training is complete type a word or letter in the training model and the model will auto complete with its own few words. 

Note: Edit the hyperparameters on the code (line 8 to 12) and Temperature in line 268, retrain the model "Python train.py" to observe how it behaves under different settings. 
For any errors or to understand the code better use Claude Code, Gemini or Codex to help you debug. 

---

## Why Does This Help You Understand Hyperparameters?

In machine learning, **hyperparameters** are like the dials and knobs on an oven — temperature, cooking time, and rack height. The AI doesn't learn these values on its own; **you** have to choose them before cooking begins. 

Most tutorials explain hyperparameters using dense math and abstract loss curves. Kitchen-GPT helps you build **intuitive gut feeling** through cause-and-effect:

**Note:** This is a toy model with very tiny data set, real word models are trained on very large set of data, so the changed in behavior of this toy model may not reflect real world scenarios in model training. 

### 1. Batch Size (How many flashcards at once?)
* **Layman analogy:** Imagine studying for an exam using flashcards. Do you review 1 card at a time or stack 16 cards together?
* **In the app:** 
  * A **small batch** trains faster, but the model learns frantically and erratically from noisy glimpses of the text.
  * A **large batch** gives the model a steadier, more balanced view of the data, making training smoother.

### 2. Block Size / Context Window (How good is the model's short-term memory?)
* **Layman analogy:** How many words or letters can you hold in your head before guessing the next word?
* **In the app:** 
  * If the block size is too **small**, the model can only remember a few letters back — it forgets the start of a word while typing the end.
  * If it's **larger**, it can recognize full cooking terms and sentence structures, but requires more training iterations to learn longer patterns effectively.

### 3. Training Iterations (Practice makes perfect vs. memorization)
* **Layman analogy:** How many times do you practice baking a recipe?
* **In the app:**
  * **Too few iterations (Underfitting):** The model produces random gibberish like an apprentice who hasn't read the recipe yet.
  * **Just right:** The model learns grammar rules, common words, and cooking concepts.
  * **Too many iterations (Overfitting):** The model stops thinking and simply regurgitates the exact training sentences verbatim.

### 4. Temperature (Creativity vs. Strict Predictability)
* **Layman analogy:** How adventurous is the chef in the kitchen?
* **In the app:**
  * **Low Temperature (0.1 – 0.3):** A cautious chef. The model always picks the safest, highest-probability next letter. The output is predictable, repetitive, but grammatically safe.
  * **Balanced (0.5 – 0.8):** A balanced cook. Coherent English with natural variety.
  * **High Temperature (1.5+):** A chaotic kitchen mad scientist. The model takes wild risks on unlikely letters, producing creative nonsense or kitchen hallucinations.

---

## The "Aha!" Moment

Instead of wondering:
> *"Why did changing the batch size or context window ruin my model's output?"*

Kitchen-GPT lets you **see and feel** the consequence:
1. Turn `block_size` all the way down to `10`, train it, and notice how it stutters and forgets words.
2. Turn `iterations` to `2000`, and watch it speak broken English.
3. Crank `temperature` to `2.0`, and watch it turn cooking advice into abstract kitchen jazz.
