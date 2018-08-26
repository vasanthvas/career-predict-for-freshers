import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
def read_All_CV(filename):
    text = textract.process(filename)
    return text.decode('utf-8')

#Next, we define a function to parse the documents (CVs) and save the word embeddings as follows:
def preprocess_training_data1(dir_cvs, dir_model_name):    
    dircvs = [join(dir_cvs, f) for f in listdir(dir_cvs) if isfile(join(dir_cvs, f))]
    alltext = ' '  
    for cv in dircvs:
        yd = read_All_CV(cv)
        alltext += yd + " "    
    alltext = alltext.lower()
    vector = []
    for sentence in es.parsetree(alltext, tokenize=True, lemmata=True, tags=True):
        temp = []
        for chunk in sentence.chunks:
            for word in chunk.words:
                if word.tag == 'NN' or word.tag == 'VB':
                    temp.append(word.lemma)
        vector.append(temp)
    global model
    model = Word2Vec(vector, size=200, window=5, min_count=3, workers=4)
    model.save(dir_model_name) 
# normalize vectors
    for string in m1.wv.vocab:
        model1[string]=m1.wv[string] / np.linalg.norm(m1.wv[string])
    # reduce dimensionality
    pca = decomposition.PCA(n_components=200)
    pca.fit(np.array(list(model1.values())))
    model1=pca.transform(np.array(list(model1.values())))
    i = 0
    for key, value in model1.items():
        model1[key] = model1[i] / np.linalg.norm(model1[i])
        i = i + 1
 with open(dir_pca_we_SWE, 'wb') as handle:
        pickle.dump(model1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model1
model1 = Word2Vec.load(join(APP_STATIC, "word2vec/ourModel"))
with open(join(APP_STATIC, 'word2vec/reduced_pca.pickle'), 'rb') as f:
model2 = pickle.load(f)

@app.route('/find/', methods=['GET'])
def find():  
    data = request.args.get('value')
    w2v = []
    aux = data.lower().split(" ")[0:5]
    sel = len(set(['java','sap','tester','prueba','hcm','sd','pruebas','testing']).intersection(aux))
    val = False
    if sel > 0:
        model = model1
        val = True
    else:
        model = model2
    if val:
        data = data.lower()
    for sentence in es.parsetree(data, tokenize=True, lemmata=True, tags=True):
        for chunk in sentence.chunks:
            for word in chunk.words:
                if val:
                    if word.lemma in model.wv.vocab:
                        w2v.append(model.wv[word.lemma])
                    else:
                        if word.lemma.lower() in model.wv.vocab:
                            w2v.append(model.wv[word.lemma.lower()])
                else:
                    if word.string in model.keys():
                    w2v.append(model[word.string])
                    else:
                    if word.string.lower() in model.keys():
                        w2v.append(model[word.string.lower()])
    Q_w2v = np.mean(w2v, axis=0)

    # Example of document represented by average of each document term vectors.
    dircvs = APP_STATIC + "/cvs_dir"
    dircvsd = [join(dircvs, f) for f in listdir(dircvs) if isfile(join(dircvs, f))]
    D_w2v = []
    for cv in dircvsd:
        yd = textract.process(cv).decode('utf-8')
        w2v = []
        for sentence in es.parsetree(yd.lower(), tokenize=True, lemmata=True, tags=True):
            for chunk in sentence.chunks:
                for word in chunk.words:
                    if val:
                        if word.lemma in model.wv.vocab:
                            w2v.append(model.wv[word.lemma])
                        else:
                            if word.lemma.lower() in model.wv.vocab:
                                w2v.append(model.wv[word.lemma.lower()])
                    else:
                     if word.string in model.keys():
                    w2v.append(model[word.string])
                   else:
                    if word.string.lower() in model.keys():
                        w2v.append(model[word.string.lower()])
        D_w2v.append((np.mean(w2v, axis=0),cv))

    # Make the retrieval using cosine similarity between query and document vectors.
    retrieval = []
    for i in range(len(D_w2v)):
        retrieval.append((1 - spatial.distance.cosine(Q_w2v, D_w2v[i][0]),D_w2v[i][1]))
    retrieval.sort(reverse=True)
    ret_data = {"cv1":url_for('static', filename="test/"+retrieval[0][1][retrieval[0][1].rfind('/')+1:]), "score1": str(round(retrieval[0][0], 4)), "cv2":url_for('static', filename="test/"+retrieval[1][1][retrieval[1][1].rfind('/')+1:]), "score2": str(round(retrieval[1][0], 4)),"cv3":url_for('static', filename="test/"+retrieval[2][1][retrieval[2][1].rfind('/')+1:]), "score3": str(round(retrieval[2][0], 4))   }
    return jsonify(ret_data)

