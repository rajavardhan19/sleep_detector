from flask import Flask, request, jsonify
from flask_cors import CORS  
from detector import detect_sleep

app = Flask(__name__)
CORS(app)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        if not data or "frame" not in data:
            print("❌ No frame data received!")
            return jsonify({"error": "No frame data received"}), 400
        
        frame_data = data["frame"]
        print(f"✅ Received frame! Data size: {len(frame_data)} bytes")  # ✅ Debugging

        asleep = detect_sleep(frame_data)

        return jsonify({"asleep": asleep})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
