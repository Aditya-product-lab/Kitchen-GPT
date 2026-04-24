import torch # The main PyTorch library for neural networks
import torch.nn as nn # Neural network modules (layers, loss functions, etc.)
from torch.nn import functional as F # Functional interface for operations like softmax

# -------------------------------------------------------------------
# SECTION 1: HYPERPARAMETERS (The Knobs YOU Turn)
# -------------------------------------------------------------------
batch_size = 16       # How many small chunks of text we process at once (Lowered to 16 for stability)
block_size = 32       # THE CONTEXT WINDOW: How many characters back the model can "see" (Lowered so it doesn't get confused)
max_iters = 5000      # How many times the model "studies" the data
learning_rate = 1e-3  # How fast the model adjusts its internal knobs. 
n_embd = 64           # The size of the "idea space" for each character. (Reduced from 256 to prevent "over-engineering")
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# SECTION 1.5: DATA PREPARATION AND PROCESSING
# -------------------------------------------------------------------
# Open the text file containing the cooking sentences
with open('cooking.txt', 'r', encoding='utf-8') as f:
    text = f.read() # Read the entire file into a single string

# Find every unique character used in the text (letters, spaces, punctuation)
chars = sorted(list(set(text)))
# Calculate the total number of unique characters (the vocabulary size)
vocab_size = len(chars)

# Create a dictionary mapping each character to a unique integer (e.g., 'a' -> 1)
stoi = { ch:i for i,ch in enumerate(chars) }
# Create a dictionary mapping each integer back to its character (e.g., 1 -> 'a')
itos = { i:ch for i,ch in enumerate(chars) }

# Create a function (lambda) that takes a string and returns a list of integers
encode = lambda s: [stoi[c] for c in s]
# Create a function (lambda) that takes a list of integers and returns a string
decode = lambda l: ''.join([itos[i] for i in l])

# -------------------------------------------------------------------
# SECTION 2: THE SELF-ATTENTION HEAD (The "Eyes" / Context)
# -------------------------------------------------------------------
class SelfAttentionHead(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__() # Initialize the parent nn.Module class
        # PARAMETERS: Linear layer to generate the "Key" (What do I look like?)
        self.key = nn.Linear(n_embd, head_size, bias=False)   
        # PARAMETERS: Linear layer to generate the "Query" (What am I looking for?)
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        # PARAMETERS: Linear layer to generate the "Value" (What information do I hold?)
        self.value = nn.Linear(n_embd, head_size, bias=False) 
        
        # Create a triangular mask matrix (1s on bottom left, 0s on top right)
        # We use register_buffer so PyTorch knows this isn't a knob to be trained, just a static tool
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Extract Batch size, Time (sequence length), and Channels (embedding size)
        B, T, C = x.shape 
        
        # Pass the input through the Linear layers to get Keys and Queries
        k = self.key(x)   # Shape: (Batch, Time, head_size)
        q = self.query(x) # Shape: (Batch, Time, head_size)
        
        # Calculate Attention Scores by multiplying Queries by Keys (and scaling down by square root of C)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # Shape: (Batch, Time, Time)
        
        # Apply the mask: replace all 0s in the upper triangle with negative infinity
        # This stops characters from "looking ahead" at characters that come after them
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        # Apply Softmax to turn the raw scores into percentages/probabilities that add up to 1.0
        wei = F.softmax(wei, dim=-1) 
        
        # Pass the input through the Value layer
        v = self.value(x) # Shape: (Batch, Time, head_size)
        
        # Multiply the attention probabilities by the Values to get the final output
        out = wei @ v     # Shape: (Batch, Time, head_size)
        return out

# -------------------------------------------------------------------
# SECTION 2.5: THE FEED-FORWARD LAYER (The "Brain" / Thinking) -> NEW!
# -------------------------------------------------------------------
class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__() # Initialize parent class
        # nn.Sequential chains multiple operations together in order
        self.net = nn.Sequential(
            # First layer expands the idea space by 4x to give it room to "think"
            nn.Linear(n_embd, 4 * n_embd),
            # ReLU (Rectified Linear Unit) is the "Activation" function. 
            # It turns negative numbers to 0, allowing the model to learn non-linear, complex patterns.
            nn.ReLU(),
            # Second layer shrinks the idea space back down to the original embedding size
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        # Pass the input through the sequential network and return the result
        return self.net(x)

# -------------------------------------------------------------------
# SECTION 3: THE TRANSFORMER ARCHITECTURE (Putting it together)
# -------------------------------------------------------------------
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__() # Initialize parent class
        # Lookup table that stores the "meaning" of each character
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Lookup table that stores the "meaning" of each position (1st letter vs 10th letter)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Initialize the Attention Head (The "Eyes")
        self.sa_head = SelfAttentionHead(n_embd)
        # Initialize the Feed-Forward Network (The "Brain") -> NEW!
        self.ffwd = FeedForward(n_embd)
        
        # The final layer that converts the internal "idea vectors" back into vocabulary guesses
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # Extract Batch and Time from the input indices
        B, T = idx.shape

        # Convert character integers into embedding vectors
        tok_emb = self.token_embedding_table(idx) # Shape: (B, T, n_embd)
        # Create position integers (0 to T-1) and convert them to embedding vectors
        pos_emb = self.position_embedding_table(torch.arange(T)) # Shape: (T, n_embd)
        
        # Add character meaning and position meaning together
        x = tok_emb + pos_emb 
        
        # Apply Attention AND add the result back to the original input (Residual Connection) -> NEW!
        x = x + self.sa_head(x)   
        # Apply Feed-Forward AND add the result back to the original input (Residual Connection) -> NEW!
        x = x + self.ffwd(x)      
        
        # Pass the processed data through the final layer to get raw character guesses
        logits = self.lm_head(x) 

        # If we are just generating text (no targets provided), skip the loss calculation
        if targets is None:
            loss = None
        else:
            # Reshape the logits and targets to fit PyTorch's cross_entropy requirements
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Calculate how wrong the predictions were compared to the actual targets
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.5, top_k=None, top_p=None):
        """ Generates new text based on a starting input (idx) """
        # Loop for however many new characters we want to generate
        for _ in range(max_new_tokens):
            # Crop the input context to match the block_size limit (prevent memory overflow)
            idx_cond = idx[:, -block_size:]
            # Get the predictions (logits) for the current context
            logits, _ = self(idx_cond)
            
            # Focus only on the prediction for the very last character in the sequence
            logits = logits[:, -1, :] 
            # Apply TEMPERATURE to make the distribution flatter (creative) or sharper (strict)
            logits = logits / temperature

            # Apply TOP-K Filtering: Keep only the top K probabilities, set the rest to negative infinity
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') 

            # Apply TOP-P (Nucleus) Filtering: Keep probabilities that add up to P%, discard the rest
            if top_p is not None:
                # Sort the logits in descending order
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # Calculate the running total (cumulative sum) of the probabilities
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Identify which tokens push the cumulative sum over the top_p threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the mask right by one so we always keep at least the #1 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter the "remove" instructions back to the original unsorted order
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # Set the discarded tokens to negative infinity
                logits[indices_to_remove] = -float('Inf')

            # Convert the final filtered/temperature-adjusted logits into probabilities (0.0 to 1.0)
            probs = F.softmax(logits, dim=-1)
            # Sample exactly 1 character based on those probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the new character to our running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        # Return the final generated sequence
        return idx

# -------------------------------------------------------------------
# SECTION 4: INITIALIZATION & TRAINING LOOP
# -------------------------------------------------------------------
# Instantiate the Transformer brain
model = MiniTransformer() 
# Create the Optimizer (The "Coach" that uses the AdamW algorithm to adjust the knobs)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Convert the entire text dataset into a single long tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

print("Training Nano-Transformer...")
# Start the main training loop
for iter in range(max_iters):
    # Pick random starting indices for our batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Stack 'batch_size' number of input sequences (length 'block_size')
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Stack the corresponding target sequences (shifted 1 character to the right)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # 1. Forward Pass: Make a guess and calculate the loss
    logits, loss = model(x, y)
    
    # 2. Reset the optimizer's gradients (clear out the notes from the last step)
    optimizer.zero_grad(set_to_none=True)
    
    # 3. Backward Pass: Calculate the derivative of the loss with respect to all parameters
    loss.backward()
    
    # 4. Optimizer Step: Slightly adjust all the internal knobs to minimize the loss
    optimizer.step()

    # Every 500 steps, print the current loss to monitor progress
    if iter % 500 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

# -------------------------------------------------------------------
# SECTION 5: INTERACTIVE CHAT MODE [OUTPUT]
# -------------------------------------------------------------------
print("\n--- Training Complete! Kitchen-GPT is ready. ---")
print("Type a few letters or a word to start. (Type 'quit' to exit)")

# Start an infinite loop to keep the chat open
while True:
    # 1. Get input from the user in the VS Code terminal
    user_prompt = input("\nYou: ")
    
    # Allow the user to exit the loop
    if user_prompt.lower() == 'quit':
        print("Shutting down Kitchen-GPT. Goodbye!")
        break
        
    # 2. The "Vocabulary Guardrail"
    # Because your model only knows the letters from 10 sentences, 
    # it will crash if you type a letter it has never seen (like 'Z' or '?').
    try:
        for char in user_prompt:
            if char not in stoi:
                raise ValueError(f"Unknown character: '{char}'")
                
        # 3. Encode the user's text into integers
        context = torch.tensor([encode(user_prompt)], dtype=torch.long)
        
        # 4. Generate the completion
        # We pass your 'context' in. The model will start from there.
        output_indices = model.generate(context, max_new_tokens=100, temperature=0.5)[0].tolist()
        
        # 5. Decode the numbers back to English and print!
        print(f"Kitchen-GPT: {decode(output_indices)}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Remember, the model only knows characters from your training data.")
        print(f"Known characters: {''.join(chars)}")