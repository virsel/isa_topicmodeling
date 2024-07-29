import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize the model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

usecase = {
    "Cleaning and Housekeeping": ["cleaning", "Housekeeping", "washing", "clothes processing", "dusting", "sanitizing"],
    "Agriculture and Husbandry": ["animal feeding", "veterinary care", "Husbandry", "livestock inspection", "poultry inspection", "pet management", "pruning", "composting", "gardening", "agriculture", "harvesting", "mowing", "tree planting", "fruit picking", "plant growth"],
    "Military": ["Army", "Combat", "military", "Weapon", "Tank", "Combat drone", "Missile", "Explosive", "Artillery", "Grenade", "Torpedo"],
    "Health and Well-being": ["medical", "Well-being", "health", "medicine", "surgical", "blind guiding", "massage", "physiotherapy", "nursing", "elderly care"],
    "Security and Rescue Services": ["security", "rescue", "home security", "protection monitoring", "fire fighting", "fire extinguishing"],
    "Kitchen Technology and Hospitality": ["kitchen", "hospitality", "cooking", "coffee making", "food processor", "meal delivering"],
    "Warehousing and Logistics": ["warehousing", "goods sorting", "logistics", "storage", "grasping and stacking"],
    "Handcraft": ["Handcraft", "measuring", "welding", "gluing", "drilling system", "tile paving", "paint spraying", "coal mine"],
    }

technology = {
    "Energy Supply and Charging Infrastructure": ["charging", "power supply", "control circuits", "cables", "battery", "energy supply"],
    "Teleoperation": ["remote communication", "opt-in request", "teleoperation", "remote control", "remote operation", "telepresence", "remote monitoring", "remote maintenance", "remote inspection"],
    "Autonomy": ["Automated Guided Vehicle", "autonomous", "autonomy", "autonomous", "self-learning", "self-optimizing", "self-healing", "self-organizing", "self-configuring", "self-adapting", "self-protecting", "self-repairing", "self-replicating", "self-sustaining"],
    "Exoskeleton": ["exoskeleton", "wearable", "assistive device", "rehabilitation", "orthosis", "prosthesis"],
    "Shape and Movement": ["body form", "body shape", "movement", "humanoid", "balance mechanism", "damping", "anti falling mechanism", "frog mechanism", "jellyfish mechanism", "amphibious", "snakelike", "spherical", "wall climbing"],
    "Physical Handcraft": ["printing instrument", "flexible gripper", "telescopic arm", "mechanical arm", "multi-angle adjustable", "suction cup", "Physical Handcraft", "grasping and stacking"],
    "Sensorial Handcraft": ['sensor data analysis', 'sensor data simulation', 'sensor data sharing', 'detection', 'laser', 'sensors', 'SLAM environment recognition and mapping', 'sensor data evaluation', 'sensor integration', 'sensor data classification', 'sensor data monitoring', 'sensor data processing', 'sensor data management', 'camera', 'sensor data interpretation', 'sensor data optimization', 'visual inspection', 'radar', 'sensor data prediction', 'sonar', 'infrared', 'thermal imaging', 'ultrasonic', 'sensor data', 'sensor data network'],
    "Material": ["radiation resistant", "material", "lightweight", "protective cover", "heat dissipation", "wear resistance", "corrosion resistance", "insulation", "waterproof", "fireproof", "anti-static", "anti-corrosion", "anti-oxidation", "anti-aging", "anti-wear", "anti-slip", "anti-fouling", "anti-UV", "anti-impact", "anti-vibration"]
}

de_usecase = {
    "Reinigung und Haushalt": ["Reinigung", "Haushalt", "Waschen", "Kleiderverarbeitung", "Staubwischen", "Desinfektion"],
    "Landwirtschaft und Tierhaltung": ["Tierfütterung", "Tierarztpflege", "Tierhaltung", "Viehinspektion", "Geflügelinspektion", "Haustiermanagement", "Beschneiden", "Kompostierung", "Gartenarbeit", "Landwirtschaft", "Ernte", "Mähen", "Baumpflanzung", "Obstpflücken", "Pflanzenwachstum"],
    "Militär": ["Armee", "Kampf", "Militär", "Waffe", "Panzer", "Kampfdrohne", "Rakete", "Sprengstoff", "Artillerie", "Granate", "Torpedo"],
    "Gesundheit und Wohlbefinden": ["Medizinisch", "Wohlbefinden", "Gesundheit", "Medizin", "Chirurgisch", "Blindenführung", "Massage", "Physiotherapie", "Krankenpflege", "Seniorenpflege"],
    "Sicherheits- und Rettungsdienste": ["Sicherheit", "Rettung", "Haussicherheit", "Schutzüberwachung", "Feuerbekämpfung", "Brandlöschung"],
    "Küchentechnologie und Gastgewerbe": ["Küche", "Gastgewerbe", "Kochen", "Kaffeezubereitung", "Lebensmittelverarbeitung", "Essenslieferung"],
    "Lagerung und Logistik": ["Etikettierung", "Lagerung", "Waren sortieren", "Logistik", "Lagerung", "Greifen und Stapeln"],
    "Handwerk": ["bauprozess", "Handwerk", "Messen", "Schweißen", "Kleben", "Bohrsystem", "Fliesenverlegung", "Farbsprühen", "Kohlebergwerk", "bauwerk"],
}

de_technology = {
    "Energieversorgung und Ladeinfrastruktur": ["Laden", "Stromversorgung", "Steuerschaltungen", "Kabel", "Batterie", "Energieversorgung"],
    "Teleoperation": ["Fernkommunikation", "Opt-in-Anfrage", "Teleoperation", "Fernsteuerung", "Fernbedienung", "Telepräsenz", "Fernüberwachung", "Fernwartung", "Ferninspektion"],
    "Autonomie": ["Automatisch geführtes Fahrzeug", "autonom", "Autonomie", "selbstlernend", "selbstoptimierend", "selbstheilend", "selbstorganisierend", "selbstkonfigurierend", "selbstanpassend", "selbstschützend", "selbstreparierend", "selbstreplizierend", "selbsterhaltend"],
    "Exoskelett": ["Exoskelett", "tragbar", "Hilfsgerät", "Rehabilitation", "Orthese", "Prothese"],
    "Form und Bewegung": ["Körperform", "Körpergestalt", "Bewegung", "humanoid", "Gleichgewichtsmechanismus", "Dämpfung", "Anti-Sturz-Mechanismus", "Froschmechanismus", "Quallenmechanismus", "amphibisch", "schlangenartig", "kugelförmig", "Wandklettern"],
    "Physisches Handwerk": ["bauen", "Druckinstrument", "flexibler Greifer", "Teleskoparm", "mechanischer Arm", "multiwinklig einstellbar", "Saugnapf", "Physisches Handwerk", "Greifen und Stapeln"],
    "Sensorisches Handwerk": ["Bildaufnahme", 'Sensor', 'Erkennung', 'Laser', 'Sensor-Datenkommunikation', 'SLAM-Umgebungserkennung und -Kartierung', 'Sensor-Datenvisualisierung', 'Sensor-Integration', 'Sensor-Datenklassifikation', 'Kamera', 'Sensor-Datenintegration', 'Sensorfusion', 'visuelle Inspektion', 'Radar',  'Sonar', 'Infrarot', 'Sensor-Datenprognose', 'Sensor-Datenabruf', 'Wärmebild', 'Ultraschall'],
    "Material": ["strahlungsbeständig", "Material", "leicht", "Schutzabdeckung", "Wärmeableitung", "Verschleißfestigkeit", "Korrosionsbeständigkeit", "Isolierung", "wasserdicht", "feuerfest", "antistatisch", "korrosionsbeständig", "oxidationsbeständig", "alterungsbeständig", "verschleißfest", "rutschfest", "schmutzabweisend", "UV-beständig", "stoßfest", "vibrationsbeständig"]
}

technology_embeddings = {}
usecase_embeddings = {}

for topic, keywords in technology.items():
    technology_embeddings[topic] = []
    technology_embeddings[topic] = model.encode(keywords, convert_to_tensor=True).to(torch.float32)
    
for topic, keywords in usecase.items():
    usecase_embeddings[topic] = []
    usecase_embeddings[topic] = model.encode(keywords, convert_to_tensor=True).to(torch.float32)

de_technology_embeddings = {}
de_tasks_embeddings = {}

for topic, keywords in de_technology.items():
    de_technology_embeddings[topic] = []
    de_technology_embeddings[topic] = model.encode(keywords, convert_to_tensor=True).to(torch.float32)

for topic, keywords in de_usecase.items():
    de_tasks_embeddings[topic] = []
    de_tasks_embeddings[topic] = model.encode(keywords, convert_to_tensor=True).to(torch.float32)


technology_boundary = 0.1
de_technology_boundary = 0.06
usecase_boundary = 0.11
de_usecase_boundary = 0.07

# Function to assign topics to a df patent row
def get_topics(row, technical=True):
    doc_embedding = row['emb']
    lang_de = row['lang'] == 'de'
    if lang_de:
        topics = de_tasks_embeddings if not technical else de_technology_embeddings
        boundary = de_technology_boundary if technical else de_usecase_boundary
    else:
        topics = usecase_embeddings if not technical else technology_embeddings
        boundary = technology_boundary if technical else usecase_boundary
        
    topic_scores = compute_topic_scores(doc_embedding, topics)

    # Sort the topics based on the scores
    sorted_topics = sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)

    top_topic_score = sorted_topics[0][1]
    if top_topic_score < boundary:
        return []
    
    topic_keys = list(topics.keys())
    # set top topic as the first topic
    topics = [topic_keys.index(sorted_topics[0][0])]
    
    # return all topics which have similarity score within 0.01 of the top topic
    for topic, score in sorted_topics[1:]:
        if top_topic_score - score < 0.015:
            topics.append(topic_keys.index(topic))
        else:
            break
        
    return topics

# Function to compute average similarity score for each topic based on top 2 keyword similarities
def compute_topic_scores(doc_embedding, topics):
    topic_scores = {}
    for topic, keyword_embeddings in topics.items():
        # Compute cosine similarity
        cos_scores = util.pytorch_cos_sim(doc_embedding, keyword_embeddings).cpu().numpy()
        
        # Flatten the cos_scores array if necessary
        cos_scores_flat = cos_scores.flatten()

        # Get the top 2 cosine similarity scores
        top_scores = np.sort(cos_scores_flat)[-2:]

        # Calculate the mean of the top 2 scores
        mean_scores = np.mean(top_scores)
        
        topic_scores[topic] = mean_scores
        
    return topic_scores

