from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from openai import OpenAI
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///golf_caddy.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

client = OpenAI(api_key='sk-proj-1Zatllzfe1VSwU0JGSKbT3BlbkFJ9KofsMF67lK3bHqn8vj0')

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or get a collection
collection_name = 'golf_caddy_vectors'
collection = chroma_client.get_or_create_collection(name=collection_name)

# Database models
class PreRoundInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), unique=True, nullable=False)
    player_name = db.Column(db.String(80), nullable=False)
    handicap = db.Column(db.Integer, nullable=False)
    driver_distance = db.Column(db.Integer, nullable=False)
    iron_7_distance = db.Column(db.Integer, nullable=False)
    characteristics = db.Column(db.String(200), nullable=False)
    vector_representation = db.Column(db.PickleType, nullable=True)  # Store the vector representation

with app.app_context():
    db.create_all()

vectorizer = TfidfVectorizer()

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Golf Caddy API"})

@app.route('/save_pre_round_info', methods=['POST'])
def save_pre_round_info():
    data = request.json
    user_id = data.get('user_id')
    player_name = data.get('player_name')
    handicap = data.get('handicap')
    driver_distance = data.get('club_distances').get('driver')
    iron_7_distance = data.get('club_distances').get('iron_7')
    characteristics = data.get('characteristics')

    pre_round_info = PreRoundInfo(
        user_id=user_id,
        player_name=player_name,
        handicap=handicap,
        driver_distance=driver_distance,
        iron_7_distance=iron_7_distance,
        characteristics=characteristics
    )

    # Create vector representation
    description = f"{player_name} {handicap} {driver_distance} {iron_7_distance} {characteristics}"
    vector_representation = vectorizer.fit_transform([description]).toarray()[0]
    pre_round_info.vector_representation = vector_representation

    db.session.add(pre_round_info)
    db.session.commit()

    # Save vector to ChromaDB
    collection.upsert(
        documents=[description],
        ids=[user_id]
    )

    return jsonify({"message": "Pre-round info saved successfully"}), 201

@app.route('/get_pre_round_info/<user_id>', methods=['GET'])
def get_pre_round_info(user_id):
    data = PreRoundInfo.query.filter_by(user_id=user_id).first()
    if data:
        response = {
            "user_id": data.user_id,
            "player_name": data.player_name,
            "handicap": data.handicap,
            "club_distances": {
                "driver": data.driver_distance,
                "iron_7": data.iron_7_distance
            },
            "characteristics": data.characteristics
        }
        return jsonify(response), 200
    else:
        return jsonify({}), 404

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    user_input = data.get('input')

    # Fetch pre-round info for the user
    user_info = PreRoundInfo.query.filter_by(user_id=user_id).first()
    if user_info:
        user_info_dict = {
            "user_id": user_info.user_id,
            "player_name": user_info.player_name,
            "handicap": user_info.handicap,
            "club_distances": {
                "driver": user_info.driver_distance,
                "iron_7": user_info.iron_7_distance
            },
            "characteristics": user_info.characteristics
        }

        # Prepare the prompt for GPT-3.5-turbo
        messages = [
            {"role": "system", "content": "You are a helpful golf caddy."},
            {"role": "user", "content": f"Player Info: {user_info_dict}"},
            {"role": "user", "content": f"User Input: {user_input}"}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )

        return jsonify({"response": response.choices[0].message.content.strip()}), 200
    else:
        return jsonify({"error": "User not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
