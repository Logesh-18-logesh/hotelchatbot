import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and tokenizer from the saved directory
model = GPT2LMHeadModel.from_pretrained('./hotel-booking-model')
tokenizer = BertTokenizer.from_pretrained('./hotel-booking-model')

# Set the model to evaluation mode
model.eval()

# Function to generate a response and stop at the first question
def generate_response(prompt, max_length=150):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a response using the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated response back to text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the response and return only the first sentence or question
    first_question = response.split('?')[0] + '?'
    return first_question

# Flask route to handle chatbot interaction
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')  # Get the message from the frontend
    bot_response = generate_response(user_message)  # Generate bot response using model
    return jsonify({'response': bot_response})  # Return the response as JSON

if __name__ == '__main__':
    app.run(debug=True)
