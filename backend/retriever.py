class Retriever:
    def __init__(self):
        self.chunks = [
            {"text": "Alexander Graham Bell was a Scottish-American inventor credited with inventing and patenting the first practical telephone in 1876.", "topic": "Alexander Graham Bell"},
            {"text": "Albert Einstein developed the theory of relativity. He received the Nobel Prize in Physics in 1921. Einstein was born on 14 March 1879.", "topic": "Albert Einstein"},
            {"text": "Isaac Newton formulated the laws of motion and universal gravitation. He was born on 4 January 1643.", "topic": "Isaac Newton"},
            {"text": "World War II lasted from 1939 to 1945. Germany invaded Poland on 1 September 1939. The war ended in 1945 with the victory of the Allies.", "topic": "World War II"},
            {"text": "The RMS Titanic sank on 15 April 1912 after striking an iceberg. More than 1,500 people died.", "topic": "Titanic"},
            {"text": "Neil Armstrong became the first person to walk on the Moon on July 20, 1969, during the Apollo 11 mission.", "topic": "Moon landing"},
            {"text": "Mahatma Gandhi led India's independence movement using nonviolent resistance. India gained independence on 15 August 1947.", "topic": "Mahatma Gandhi"},
            {"text": "DNA carries genetic instructions for all known organisms. The structure of DNA was discovered by James Watson and Francis Crick in 1953.", "topic": "DNA"},
            {"text": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.", "topic": "Python"},
            {"text": "Marie Curie was the first woman to win a Nobel Prize. She won the Nobel Prize twice - Physics in 1903 and Chemistry in 1911.", "topic": "Marie Curie"},
        ]
        print(f"Retriever loaded! {len(self.chunks)} documents")

    def retrieve(self, question: str, top_k: int = 3):
        question_lower = question.lower()
        scored = []
        
        for chunk in self.chunks:
            score = 0
            topic_words = chunk['topic'].lower().split()
            text_words = chunk['text'].lower().split()
            
            for word in question_lower.split():
                if len(word) > 3:
                    if word in topic_words:
                        score += 3
                    if word in text_words:
                        score += 1
            
            scored.append({
                'text': chunk['text'],
                'topic': chunk['topic'],
                'score': score
            })
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]