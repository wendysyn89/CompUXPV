# CompUXPV
This is an user experience prediction engine using computational semantics approaches- Paragraph Vector by Quoc Le. 

Files usages:

review.txt -> Review for prediction
all_item_label.txt -> Total of 1028 items (5 construct category) for prediction. Last line (1029) is default value to be replaced with review's vector during prediction.
The prediction engine will derive paragraph vector for both review and measurement items, next will compute their semantic similarity.

CompUXPV.py -> main file for UX prediction engine. Able to learn word vector, paragraph vector (review and items), and predict the most suitable UX.
In this example, we have 1028 items that comprised of 5 UX construct category (Perceived usefulness, perceived ease of use, affects towards technology, social influence, trust)


