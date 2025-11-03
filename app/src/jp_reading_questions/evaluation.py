evaluation_dataset = [
    {
        "inputs": {"jp_text": "今日は天気が良いので、公園で散歩をしました。桜の花がとてもきれいでした。"},
        "expectations": {
            "expected_questions": [
                {
                    "category": "事実",
                    "question": "話者は今日何をしましたか？",
                    "options": ["A. 公園で散歩をした", "B. 桜を植えた", "C. 家で休んだ", "D. 買い物に行った"],
                    "answer": "A"
                }
            ]
        },
    },
    {
        "inputs": {"jp_text": "彼女は毎朝6時に起きて、ジョギングをする習慣があります。健康を保つために大切だと考えているからです。"},
        "expectations": {
            "expected_questions": [
                {
                    "category": "暗示されたメッセージ",
                    "question": "この文から分かる彼女の考えは何ですか？",
                    "options": ["A. 早起きは難しい", "B. 運動は健康に重要", "C. ジョギングは楽しい", "D. 朝は忙しい"],
                    "answer": "B"
                }
            ]
        },
    },
]
