import discord
import os
from dotenv import load_dotenv
from discord.ext import commands
import tensorflow as tf
import numpy as np
import pickle

load_dotenv()
token = os.getenv('TOKEN')

# Define the intents your bot needs
intents = discord.Intents.default()
intents.message_content = True # Enable bot to receive message content

# Create the bot instance with the defined intents
bot = commands.Bot(command_prefix='!', intents=intents) 
# You can change the command prefix to whatever you want

loaded_model = tf.keras.models.load_model('lstm_model.keras')
max_len = 100 # max length of each text (in terms of number of words)
#tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.command()
async def hello(ctx):
    await ctx.send("Hello!")

"""
@bot.command()
async def predict(ctx):
    try:
        input_value = float(ctx.message.content.split()[1])
        prediction = loaded_model.predict(np.array([[input_value]]))[0][0]
        print(f"prediction: {prediction}, input_value: {input_value}")
        
        await ctx.send(f"Prediction for {input_value}: {prediction:.2f}")
    except:
        await ctx.send("Invalid Input! Please use !predict [number]")
"""
@bot.command()
async def predict(ctx):
    try:

        def predict_sentiment(text, loaded_model, tokenizer, max_len):
            """Predicts the sentiment of a given text."""

            # 1. Preprocess the text
            sequence = tokenizer.texts_to_sequences([text])  # Tokenize
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                sequence, maxlen=max_len
            )  # Pad

            # 2. Make predictions
            predictions = loaded_model.predict(padded_sequence)

            # 3. Interpret the results (binary classification)
            positive_probability = predictions[0][1]  # Assuming one output neuron
            #negative_probability = predictions[0][0]  # Assuming one output neuron

            if positive_probability >= 0.5:
                return "Positive", positive_probability, predictions
            else:
                return "Negative", positive_probability, predictions
        
        parts = ctx.message.content.split(":", 1)
        if len(parts) != 2:
            raise ValueError("Input must contain ':'")

        input_value = parts[1].strip()  # Extract the string after ':' and remove leading/trailing spaces.

        new_text = input_value
        sentiment, probability, prediction = predict_sentiment(new_text, loaded_model, tokenizer, max_len)


        # 4. Send the result
        await ctx.send(f"Prediction for '{input_value}': Sentiment: {sentiment}, Probability: {probability}")
        #await ctx.send(f"Prediction for '{input_value}': Sentiment: {sentiment}, Probability: {positive_probability} (Predictions: {predictions:.2f})")

    except ValueError as e:
        await ctx.send(f"Invalid Input! Please use !predict : [text]. Error: {e}")
    except Exception as e:
        await ctx.send(f"An unexpected error occurred: {e}")

# Run the bot
bot.run(token)