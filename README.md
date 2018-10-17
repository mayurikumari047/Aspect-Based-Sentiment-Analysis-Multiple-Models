# Aspect-Based-Sentiment-Analysis
Aspect based sentiment analysis is the determination of sentiment orientation of different important aspects of any textual review or post. 

This research work is done on Aspect Based Sentiment Analysis concentrating on aspect term polarity estimation for given textual dataset. The dataset consists of a number of examples/reviews and each example/review is paired with an aspect its location in the sentence for which the sentiment polarity needs to be estimated. 

Used multiple machine learning models for the determination of the sentiment polarity in reviews/post based on aspect and presented their pros and cons. 
Applied various machine learning techniques, tuned parameters on different models to achieve high accuracy. 
This project required concepts from Machine learning, Natural Language Processing, Deep Learning and Data Mining.

Technique :
Building a classifier that can predict the sentiment polarity of text based on aspect starts with data preprocessing steps, preparing the feature matrix of the data that can be fed into Deep memory network and other classifiers too. Building the Deep Memory network model and finally running the Deep memory network and other classifiers such as SVM, Random Forest and Voted Classifier. 

Success was evaluated based on three criteria: precision, recall, and F1-Measure.

Data Preprocessing:
The data needs to be cleaned and prepared before it can be fed to any machine learning model.
The steps taken in order to preprocess the data were: 
Replaced '[comma]' with actual ', lowercase all strings, Number to text, Custom tokenization with punctuation removal.

The feature matrix consists of four lists - source data, source location data, target data and maximum length among all the sentences. For preparing the feature matrix, two dictionaries were created: 
1. Source2idx - maintained all words from the text against a unique id, 
2. Target2idx - maintained all the aspect terms (as one word) against a unique id. 
The tokenized sentence was replaced with ids from these dictionaries that created the source data and target data. The next
step involved transforming the tokenized sentence into a location from aspect based list. The source location data list consisted of the distance of every word from the aspect term in the sentence. The max length of the sentence was also maintained as the last list with single element.

Classifiers:
Building the Deep memory network model
Our design of Deep memory network is based on the works of Duyu Tang , Bing Qin , Ting Liu which was published at Cornell University Library on 28 May 2016. A memory network consists of a memory m and four components I, G, O and R, where m is an array of objects such as an array of vectors. Among these four components, ‘I’ converts input to internal feature representation, G updates old memories with new input, O generates an output representation given a new input and the current memory state, R outputs a response based on the output representation. Given a sentence s = {w1, w2, ..., wi , ...wn} and the aspect word wi , we map each word into its embedding vector. These word vectors are separated into two parts, aspect representation and context representation. Aspect representation is an average of the embedding of its constituting aspect word vectors. Context word vectors {e1, e2 ... ei−1, ei+1 ... en} are stacked and regarded as the external memory m ∈ R d×(n−1) , where n is the sentence length. We assigned weights to each context word vectors by using pre- trained data from GloVe: Global Vectors for Word Representation. An illustration of this approach is shown below: This approach consists of multiple computational layers (hops), each of which contains an attention layer and a linear layer. In the first computational layer (hop 1), we regard aspect vector as the input to adaptively select important evidences from memory m through attention layer.
The output of attention layer and the linear transformation of aspect vector2 are summed and the result is considered as the input of next layer (hop 2). In a similar way, we stack multiple hops and run these steps multiple times, so that more abstractive evidences could be selected from the external memory m. The output vector in last hop is considered as the representation of sentence
with regard to the aspect, and is further used as the feature for aspect level sentiment classification. Feature vectors were split into train and test vectors and fed into this model for training, validation and testing to get the precision, recall and F-score for success evaluation. Other classification methods tried:
Same feature vectors prepared earlier which had context words index, context words location index and aspect word index were fed to multiple classifiers from Scikit-learn library and their aggregated voted prediction was considered for making final decision. We used ‘Voting Classifier’ from Scikit-learn library for this voting for classification. The classifiers used for
Voting were ‘Random Forest’, SVC (Support Vector Classifier) and Decision Tree classifiers.

Evaluation Results
I. VotingClassifier(Random Forest, Decision Tree, Linear SVM)
a) Laptop Data:
Class Label Precision Recall F1-Score
-1 0.77 0.71 0.74
0 0.77 0.80 0.78
1 0.75 0.78 0.76
Average 0.76 0.76 0.76
● Average f1-scores after 10-Fold Cross Validation:
[0.75047985 0.69807692 0.77649326 0.77456647 0.78998073 0.83622351 0.72639692
0.7495183 0.75722543 0.73217726]
● Classifier accuracy: 0. 76
b) Restaurant Data:
Class Label Precision Recall F1-Score
-1 0.78 0.75 0.75
0 0.80 0.78 0.79
1 0.74 0.80 0.77
Average 0.77 0.77 0.77
● Average f1-scores after 10-Fold Cross Validation:
[0.73320537 0.71153846 0.78420039 0.75337187 0.7822736 0.83622351 0.77456647
0.74566474 0.73603083 0.75337187]
● Distribution of f1-scores: Mean: 0.64, Maximum: 0.83, Minimum: 0.71
● Classifier accuracy: 0. 77
II. Deep Memory Neural Networks(Attention Model)
c) Laptop Data:
Class Label Precision Recall F1-Score
-1 0.58 0.60 0.59
0 0.69 0.73 0.71
1 0.63 0.57 0.60
Average 0.63 0.63 0.63
● Epochs : 5, batch_size : 128
● Classifier accuracy: 0. 63
d) Restaurant Data:
Class Label Precision Recall F1-Score
-1 0.65 0.64 0.64
0 0.68 0.68 0.68
1 0.64 0.64 0.64
Average 0.65 0.65 0.65
● Epochs : 5, batch_size : 128
● Classifier accuracy: 0. 65
Conclusion
Aspect based sentiment analysis, as interesting as it was, requires further research and in-depth
analysis to reach a high level of desired accuracy. The dataset provided had 2553 and 3340 rows
respectively. We experimented with different feature matrix building techniques and classifiers.
The SVM with poly kernel gave bas results as compared to other kernels. Random forest with
500 estimators reached to the accuracy of 76%. The decision tree model was applied and the
overall accuracy turned out to be 65%. When weighted voting classifier was applied to Random
Forest, SVM in conjunction with Decision Tree, gave 76% accuracy.
Deep neural networks behaved better and reached to an accuracy score of 63% for restaurant
dataset with merely 3 epochs and 7 hops. We plan to run 100 epochs as suggested by the
reference paper in order to increase accuracy manifolds.
References
1. scikit-learn: http://scikit-learn.org/stable
2. https://arxiv.org/pdf/1605.08900.pdf
3. http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
4. http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifi
er.html
5. http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
6. http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
