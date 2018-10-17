# Aspect-Based-Sentiment-Analysis
Aspect based sentiment analysis is the determination of sentiment orientation of different important aspects of any textual review or post. 

This research project is done on Aspect Based Sentiment Analysis concentrating on aspect term polarity estimation for given textual dataset. The dataset consists of a number of examples/reviews and each example/review is paired with an aspect its location in the sentence for which the sentiment polarity needs to be estimated. Used multiple machine learning models including Deep Memory network for the determination of the sentiment polarity in reviews/post based on aspect and presented their pros and cons. Applied various machine learning techniques, tuned parameters on different models to achieve high accuracy. 

Deep memory network approach explicitly captures the importance of each context word when inferring the sentiment polarity of an aspect. Such importance degree and text representation are calculated with multiple computational layers, each of which is a neural attention model over an external memory. Experiments on laptop and restaurant datasets demonstrates that this approach performs comparable to feature based SVM model and Random Forest ensemble method. We show that multiple computational layers can improve the performance of the system.

Aspect Based Sentiment Analysis systems receives a set of texts as input such as product reviews or messages from social media, discussing a particular entity (e.g., laptop or restaurant). It attempts to detect the main or the most frequently discussed aspects of the entity such as ‘battery’, ‘screen’ and to estimate the sentiment polarity of the texts per aspect that is whether the opinion of each aspect is positive, negative or neutral. For example in sentence ‘screen is nice but the battery is pathetic’, the sentiment polarity of aspect ‘screen’ is positive but the sentiment polarity of aspect ‘battery’ is negative.
Many researches are happening on this topic but approaches based on neural networks are the winning ones these days because of their ability to learn text representation from data and to capture semantic relations between aspect and context words in better way as compared to other model like feature based SVM.

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

Building the Deep memory network model:
Our design of Deep memory network is based on the works of Duyu Tang , Bing Qin , Ting Liu which was published at Cornell University Library on 28 May 2016. A memory network consists of a memory m and four components I, G, O and R, where m is an array of objects such as an array of vectors. Among these four components, ‘I’ converts input to internal feature representation, G updates old memories with new input, O generates an output representation given a new input and the current memory state, R outputs a response based on the output representation. Given a sentence s = {w1, w2, ..., wi , ...wn} and the aspect word wi , we map each word into its embedding vector. These word vectors are separated into two parts, aspect representation and context representation. Aspect representation is an average of the embedding of its constituting aspect word vectors. Context word vectors {e1, e2 ... ei−1, ei+1 ... en} are stacked and regarded as the external memory m ∈ R d×(n−1) , where n is the sentence length. We assigned weights to each context word vectors by using pre- trained data from GloVe: Global Vectors for Word Representation. An illustration of this approach is shown below: This approach consists of multiple computational layers (hops), each of which contains an attention layer and a linear layer. In the first computational layer (hop 1), we regard aspect vector as the input to adaptively select important evidences from memory m through attention layer.
The output of attention layer and the linear transformation of aspect vector2 are summed and the result is considered as the input of next layer (hop 2). In a similar way, we stack multiple hops and run these steps multiple times, so that more abstractive evidences could be selected from the external memory m. The output vector in last hop is considered as the representation of sentence
with regard to the aspect, and is further used as the feature for aspect level sentiment classification. Feature vectors were split into train and test vectors and fed into this model for training, validation and testing to get the precision, recall and F-score for success evaluation. 

Other classification methods:
Same feature vectors prepared earlier which had context words index, context words location index and aspect word index were fed to multiple classifiers from Scikit-learn library and their aggregated voted prediction was considered for making final decision. We used ‘Voting Classifier’ from Scikit-learn library for this voting for classification. The classifiers used for
Voting were ‘Random Forest’, SVC (Support Vector Classifier) and Decision Tree classifiers.

Conclusion
Aspect based sentiment analysis, as interesting as it was, requires further research and in-depth analysis to reach a high level of desired accuracy. The dataset provided had 2553 and 3340 rows respectively. We experimented with different feature matrix building techniques and classifiers. The SVM with poly kernel gave bas results as compared to other kernels. Random forest with 500 estimators reached to the accuracy of 78%. The decision tree model was applied and the overall accuracy turned out to be 69%. When weighted voting classifier was applied to Random Forest, SVM in conjunction with Decision Tree, gave 76% accuracy. Deep neural networks behaved better and reached to an accuracy score of 63% for restaurant dataset with merely 3 epochs and 7 hops. 

References
1. http://scikit-learn.org/stable
2. https://arxiv.org/pdf/1605.08900.pdf
3. http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
4. http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
5. http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
6. http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
