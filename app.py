from flask import Flask, request, jsonify
from recommendation import get_recommendations

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        product_id = request.args.get('product_id')
        if product_id is None:
            return jsonify({"error": "Missing product_id parameter"}), 400

        product_id = int(product_id)  # Convert to Python int
        recommendations = get_recommendations(product_id)

        if not recommendations:
            return jsonify({"message": "No recommendations found for this product"}), 404
        
        # Convert NumPy int64 to Python int
        return jsonify({"recommended_products": [int(p) for p in recommendations]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)