import torch
import os
import gzip
import argparse
import numpy as np
from exponential.algos import EMCov
from adaptive.algos import GaussCov, LapCov, SeparateCov, SeparateLapCov, AdaptiveCov, AdaptiveLapCov
from coinpress.algos import cov_est
from urllib.request import urlretrieve
from sklearn.feature_extraction.text import HashingVectorizer
from glob import glob
import re
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_budget', default=.1, type=float, help='total privacy budget (rho)')
    parser.add_argument('--d', default=200, type=int, help='data dimension/number of features')
    parser.add_argument('--n', default=50000, type=int, help='sample size')
    parser.add_argument('--r', default=1.0, type=float, help='l2 norm upperbound')
    parser.add_argument('--u', default=1.0, type=float, help='eigenvalue upperbound for coinpress')
    parser.add_argument('--beta', default=0.1, type=float, help='prob. bound')
    parser.add_argument('--s', default=3, type=float, help='steepness in Zipf law')
    parser.add_argument('--N', default=4, type=float, help='number of buckets in Zipf law')
    args = parser.parse_args()
    return args

def write_output(x,y,folder,filename):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if (len(np.array(y).shape)) == 1:
        out = [x,y]
    else:
        out = []
        for i in range(len(x)):
            out.append([x[i]])
            out[i].extend(y[i])
    np.savetxt(folder+filename,np.transpose(out))

def write_text(x,folder,filename):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fOut = open(folder+filename,'w')
    for xi in x:
        fOut.write(','.join(xi)+'\n')
    fOut.close()

def extract_news_info(filename, num_news=1000, dtype=str):
    with gzip.open(filename) as bytestream:
        buf = bytestream.read().decode()
        articles = buf.split('\n\n')
        dict_words = {}
        for article in articles:
            words = re.split('\n|;|,|:| |-',article)
            for word in words:
                word_l = word.lower()
                if word_l.isalpha():
                    if word_l in dict_words.keys():
                        dict_words[word_l] += 1
                    else:
                        dict_words[word_l] = 1
        return dict_words, articles
    
def get_news_data(d,norm="l2",b_alt=False):
    folder = './data/'
    if not os.path.exists(folder):
        os.mkdir('data')
    filenames = ['news-commentary-v16.en.gz']
    for name in filenames:
        urlretrieve('https://data.statmt.org/news-commentary/v16/training-monolingual/' + name, folder+name)
    articles = extract_data_news(folder,filenames)
    vectorizer = HashingVectorizer(n_features=d,norm=norm,alternate_sign=b_alt,dtype=np.float32)
    X = vectorizer.fit_transform(articles)
    return torch.from_numpy(X.todense())
            
def extract_data_news(folder,filenames):
    articles = []
    for filename in filenames:
        with gzip.open(folder+filename) as bytestream:
            buf = bytestream.read().decode()
            articles.extend(buf.split('\n\n'))
    bad_art = set()
    delim = r'[ ,;:!\n"?!-]'
    for i in range(len(articles)):
        words = re.split(delim,articles[i])
        m = len(words)
        if m < 20:
            bad_art.add(i)
    articles = [articles[j] for j in range(len(articles)) if j not in bad_art]
    return articles
             
        
def extract_data(filename, num_images, dtype=np.float32):
    d = 28*28
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*d)
        data = np.frombuffer(buf, dtype=np.uint8).astype(dtype)
        data = (data / 255) / 28
        data = data.reshape(num_images, d)
        return data
     
def get_mnist_data():
    if not os.path.exists('data'):
        os.mkdir('data')
    filenames = ["train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"]
    for name in filenames:
        urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)
    train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
    test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
    return train_data, test_data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

def get_mnist_labels():
    if not os.path.exists('data'):
        os.mkdir('data')
    filenames = ["train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for name in filenames:
        if not os.path.exists("data/"+name):
            urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)
    train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
    test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
    return train_labels, test_labels

def gen_synthetic_data_fix(d, n, s, N, seed=0):
    torch.manual_seed(seed)
    X = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(d),torch.eye(d)).sample((n,))
    U = torch.distributions.uniform.Uniform(torch.tensor([0.0 for i in range(d)]), torch.tensor([1.0 for i in range(d)])).sample((d,))
    X = torch.mm(X,U)
    mu = torch.mean(X, dim=0)
    X = X - mu
    if (N>0):
        probs, buckets = get_zipf_buckets(s, N)
        X = adjust_weight_fix(X,probs,buckets,N,n)
    return X

def adjust_weight_fix(X,probs,buckets,N,n):
    x_norm = torch.norm(X,dim=1)
    scale = torch.zeros(n)
    count_range = [0]
    count_range.extend([int(n*probs[k]) for k in range(N-1)])
    count_range.append(n)
    for k in range(N):
        l = count_range[k]
        r = count_range[k+1]
        scale[l:r] = x_norm[l:r]/buckets[k]
    return torch.div(X.t(),scale).t()
    
def get_zipf_buckets(s, N):
    numer = [1./((k+1)**s) for k in range(N)]
    denom = sum(numer)
    probs = []
    prob = 0
    for k in range(N):
        prob = prob + numer[k]/denom
        probs.append(prob)
    buckets = [2**(k+1-N) for k in range(N)]
    return probs, buckets


def test_news_rho(args,rhos,strfolder,params,norm="l1",b_alt=True,scale=1):
    d = args.d
    n = args.n
            
    err_em_paths = []
    err_gauss_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    err_cpt1_paths = []
    err_cpt2_paths = []
    err_cpt3_paths = []
    err_cpt4_paths = []
    err_cpt5_paths = []

    for j in range(len(rhos)):
        rho = rhos[j]
        args.total_budget = rho
        Ps1 = [args.total_budget]
        Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]

        X = get_news_data(d,norm=norm,b_alt=b_alt)
        if (scale=='max'):
            x_norm = torch.norm(X,dim=1,p=2)
            adj = max(x_norm)
        else:
            adj = scale
        X = X/adj
        n1,d1 = X.shape
        assert(n1==n)
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=False,b_fleig=True)
        cov_gauss = GaussCov(X.clone(),n,d,rho,b_fleig=True)
        cov_sep = SeparateCov(X.clone(),n,d,rho,b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(),args)
        
        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_gauss_paths.append(torch.norm(cov-cov_gauss,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        err_cpt1_paths.append(torch.norm(cov-cov_cpt1,'fro'))
        err_cpt2_paths.append(torch.norm(cov-cov_cpt2,'fro'))
        err_cpt3_paths.append(torch.norm(cov-cov_cpt3,'fro'))
        err_cpt4_paths.append(torch.norm(cov-cov_cpt4,'fro'))
        err_cpt5_paths.append(torch.norm(cov-cov_cpt5,'fro'))
        
    write_output(rhos,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(rhos,err_gauss_paths,strfolder,'err_gauss_paths.txt')
    write_output(rhos,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(rhos,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(rhos,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_output(rhos,err_cpt1_paths,strfolder,'err_cpt1_paths.txt')
    write_output(rhos,err_cpt2_paths,strfolder,'err_cpt2_paths.txt')
    write_output(rhos,err_cpt3_paths,strfolder,'err_cpt3_paths.txt')
    write_output(rhos,err_cpt4_paths,strfolder,'err_cpt4_paths.txt')
    write_output(rhos,err_cpt5_paths,strfolder,'err_cpt5_paths.txt')
    write_text(params,strfolder,'params.txt') 
    

def test_news_eps(args,epss,strfolder,params,norm="l1",b_alt=True,scale=1):
    d = args.d
    n = args.n
            
    err_em_paths = []
    err_lap_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []

    for j in range(len(epss)):
        eps = epss[j]
        rho = eps*eps/2.
        args.total_budget = rho
        
        X = get_news_data(d,norm=norm,b_alt=b_alt)
        if (scale=='max'):
            x_norm = torch.norm(X,dim=1,p=2)
            adj = max(x_norm)
        else:
            adj = scale
        X = X/adj
        n1,d1 = X.shape
        assert(n1==n)
        cov = torch.mm(X.t(),X)/n
        print('trace: ', float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=True,b_fleig=True)
        cov_lap = LapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_sep = SeparateLapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_adapt = AdaptiveLapCov(X.clone(),args)
        
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_lap_paths.append(torch.norm(cov-cov_lap,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        
    write_output(epss,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(epss,err_lap_paths,strfolder,'err_lap_paths.txt')
    write_output(epss,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(epss,err_adapt_paths,strfolder,'err_adapt_paths.txt') 
    write_output(epss,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_text(params,strfolder,'params.txt') 


def test_minist_fix_digit(args,rhos,strfolder,params,digit):
    train_data, test_data = get_mnist_data()
    train_labels, test_labels = get_mnist_labels()
    ind = (train_labels==digit)
    d = args.d
    n = sum(ind)
    args.n = n
            
    err_em_paths = []
    err_gauss_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    err_cpt1_paths = []
    err_cpt2_paths = []
    err_cpt3_paths = []
    err_cpt4_paths = []
    err_cpt5_paths = []

    Y = torch.from_numpy(train_data)
    X = Y[ind]
    n1,d1 = X.shape
    assert(n1==n and d1==d)
    for j in range(len(rhos)):
        rho = rhos[j]
        args.total_budget = rho
        Ps1 = [args.total_budget]
        Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)),'; digit: ',digit)
        cov_em = EMCov(X.clone(),args,b_budget=False,b_fleig=True)
        cov_gauss = GaussCov(X.clone(),n,d,rho,b_fleig=True)
        cov_sep = SeparateCov(X.clone(),n,d,rho,b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(),args)
        
        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_gauss_paths.append(torch.norm(cov-cov_gauss,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        err_cpt1_paths.append(torch.norm(cov-cov_cpt1,'fro'))
        err_cpt2_paths.append(torch.norm(cov-cov_cpt2,'fro'))
        err_cpt3_paths.append(torch.norm(cov-cov_cpt3,'fro'))
        err_cpt4_paths.append(torch.norm(cov-cov_cpt4,'fro'))
        err_cpt5_paths.append(torch.norm(cov-cov_cpt5,'fro'))
    write_output(rhos,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(rhos,err_gauss_paths,strfolder,'err_gauss_paths.txt')
    write_output(rhos,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(rhos,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(rhos,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_output(rhos,err_cpt1_paths,strfolder,'err_cpt1_paths.txt')
    write_output(rhos,err_cpt2_paths,strfolder,'err_cpt2_paths.txt')
    write_output(rhos,err_cpt3_paths,strfolder,'err_cpt3_paths.txt')
    write_output(rhos,err_cpt4_paths,strfolder,'err_cpt4_paths.txt')
    write_output(rhos,err_cpt5_paths,strfolder,'err_cpt5_paths.txt')
    write_text(params,strfolder,'params.txt') 
    
 
def test_minist_fix_n(args,rhos,strfolder,params,seeds):
    train_data, test_data = get_mnist_data()
    d = args.d
    n = args.n
            
    err_em_paths = []
    err_gauss_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    err_cpt1_paths = []
    err_cpt2_paths = []
    err_cpt3_paths = []
    err_cpt4_paths = []
    err_cpt5_paths = []

    Y = torch.from_numpy(train_data)
    n0,d0 = Y.shape
    assert(d==d0)
    if (n < n0):
        torch.manual_seed(seeds[0])
        ind = random.sample(range(0,n0),n)
        X = Y[ind]
    else:
        X = Y.clone()
    for j in range(len(rhos)):
        rho = rhos[j]
        args.total_budget = rho
        Ps1 = [args.total_budget]
        Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=False,b_fleig=True)
        cov_gauss = GaussCov(X.clone(),n,d,rho,b_fleig=True)
        cov_sep = SeparateCov(X.clone(),n,d,rho,b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(),args)
        
        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_gauss_paths.append(torch.norm(cov-cov_gauss,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        err_cpt1_paths.append(torch.norm(cov-cov_cpt1,'fro'))
        err_cpt2_paths.append(torch.norm(cov-cov_cpt2,'fro'))
        err_cpt3_paths.append(torch.norm(cov-cov_cpt3,'fro'))
        err_cpt4_paths.append(torch.norm(cov-cov_cpt4,'fro'))
        err_cpt5_paths.append(torch.norm(cov-cov_cpt5,'fro'))
    write_output(rhos,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(rhos,err_gauss_paths,strfolder,'err_gauss_paths.txt')
    write_output(rhos,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(rhos,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(rhos,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_output(rhos,err_cpt1_paths,strfolder,'err_cpt1_paths.txt')
    write_output(rhos,err_cpt2_paths,strfolder,'err_cpt2_paths.txt')
    write_output(rhos,err_cpt3_paths,strfolder,'err_cpt3_paths.txt')
    write_output(rhos,err_cpt4_paths,strfolder,'err_cpt4_paths.txt')
    write_output(rhos,err_cpt5_paths,strfolder,'err_cpt5_paths.txt')
    write_text(params,strfolder,'params.txt')  
    
    
def test_minist_fix_n_pure(args,epss,strfolder,params,seeds):
    train_data, test_data = get_mnist_data()
    d = args.d
    n = args.n
            
    err_em_paths = []
    err_lap_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []

    Y = torch.from_numpy(train_data)
    n0,d0 = Y.shape
    assert(d==d0)
    if (n < n0):
        torch.manual_seed(seeds[0])
        ind = random.sample(range(0,n0),n)
        X = Y[ind]
    else:
        X = Y.clone()
    for j in range(len(epss)):
        eps = epss[j]
        rho = eps*eps/2.
        args.total_budget = rho

        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=True,b_fleig=True)
        cov_lap = LapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_sep = SeparateLapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_adapt = AdaptiveLapCov(X.clone(),args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_lap_paths.append(torch.norm(cov-cov_lap,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        
    write_output(epss,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(epss,err_lap_paths,strfolder,'err_lap_paths.txt')
    write_output(epss,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(epss,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(epss,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_text(params,strfolder,'params.txt')     
    
    
def test_mnist_fix_pure(args,rhos,strfolder,params):
    train_data, test_data = get_mnist_data()
    d = args.d
    n = args.n
    err_em_paths = []
    err_lap_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    epss = []
    X = torch.from_numpy(train_data[:n])
    for j in range(len(rhos)):
        rho = rhos[j]
        args.total_budget = rho
        eps = np.sqrt(2*rho)
        epss.append(eps)
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=True,b_fleig=True)
        cov_lap = LapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_sep = SeparateLapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_adapt = AdaptiveLapCov(X.clone(),args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_lap_paths.append(torch.norm(cov-cov_lap,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
    write_output(epss,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(epss,err_lap_paths,strfolder,'err_lap_paths.txt')
    write_output(epss,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(epss,err_adapt_paths,strfolder,'err_adapt_paths.txt') 
    write_output(epss,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_text(params,strfolder,'params.txt')         
    
def test_n_pure(args,ns,strfolder,params,seeds):
    d = args.d
    s = args.s
    N = args.N
    rho = args.total_budget
    eps = np.sqrt(2*rho)
    err_em_paths = []
    err_lap_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    for j in  range(len(ns)):
        n = int(ns[j])
        args.n = n
        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ', float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=True,b_fleig=True)
        cov_lap = LapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_sep = SeparateLapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_adapt = AdaptiveLapCov(X.clone(),args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_lap_paths.append(torch.norm(cov-cov_lap,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
    write_output(ns,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(ns,err_lap_paths,strfolder,'err_lap_paths.txt')
    write_output(ns,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(ns,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(ns,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_text(params,strfolder,'params.txt')       


def test_d_pure(args,ds,strfolder,params,seeds):
    n = args.n
    s = args.s
    N = args.N
    rho = args.total_budget
    eps = np.sqrt(2*rho)
    err_em_paths = []
    err_lap_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    for j in  range(len(ds)):
        d = int(ds[j])
        args.d = d
        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        #cov_em = EMCov(X.clone(),args,b_budget=True,b_fleig=True)
        cov_lap = LapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_sep = SeparateLapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_adapt = AdaptiveLapCov(X.clone(),args)
        #err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_lap_paths.append(torch.norm(cov-cov_lap,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
    #write_output(ds,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(ds,err_lap_paths,strfolder,'err_lap_paths.txt')
    write_output(ds,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(ds,err_adapt_paths,strfolder,'err_adapt_paths.txt')  
    write_output(ds,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_text(params,strfolder,'params.txt')                   
            

def test_eps_pure(args,epss,strfolder,params,seeds):
    d = args.d
    n = args.n
    s = args.s
    N = args.N
    err_em_paths = []
    err_lap_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    for j in range(len(epss)):
        eps = epss[j]
        rho = eps*eps/2.
        args.total_budget = rho
        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=True,b_fleig=True)
        # cov_lap = LapCov(X.clone(),n,d,eps,b_fleig=True)
        # cov_sep = SeparateLapCov(X.clone(),n,d,eps,b_fleig=True)
        # cov_adapt = AdaptiveLapCov(X.clone(),args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        # err_lap_paths.append(torch.norm(cov-cov_lap,'fro'))
        # err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        # err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        # err_zero_paths.append(torch.norm(cov,'fro'))
    write_output(epss,err_em_paths,strfolder,'err_em_paths.txt')
    # write_output(epss,err_lap_paths,strfolder,'err_lap_paths.txt')
    # write_output(epss,err_sep_paths,strfolder,'err_sep_paths.txt')
    # write_output(epss,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    # write_output(epss,err_zero_paths,strfolder,'err_zero_paths.txt')
    # write_text(params,strfolder,'params.txt')       
      
    
def test_Ns_pure(args,Ns,strfolder,params,seeds):
    n = args.n
    d = args.d
    s = args.s
    rho = args.total_budget
    eps = np.sqrt(2*rho)
    err_em_paths = []
    err_lap_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    for j in  range(len(Ns)):
        N = int(Ns[j])
        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=True,b_fleig=True)
        cov_lap = LapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_sep = SeparateLapCov(X.clone(),n,d,eps,b_fleig=True)
        cov_adapt = AdaptiveLapCov(X.clone(),args)
        cov_adapt = AdaptiveLapCov(X.clone(),args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_lap_paths.append(torch.norm(cov-cov_lap,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
    write_output(Ns,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(Ns,err_lap_paths,strfolder,'err_lap_paths.txt')
    write_output(Ns,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(Ns,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(Ns,err_zero_paths,strfolder,'err_zero_paths.txt') 
    write_text(params,strfolder,'params.txt')   
    
    
def test_n(args,ns,strfolder,params,seeds):
    d = args.d
    s = args.s
    N = args.N
    rho = args.total_budget
    
    Ps1 = [args.total_budget]
    Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    err_em_paths = []
    err_gauss_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    err_cpt1_paths = []
    err_cpt2_paths = []
    err_cpt3_paths = []
    err_cpt4_paths = []
    err_cpt5_paths = []
    for j in  range(len(ns)):
        n = int(ns[j])
        args.n = n
        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ', float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=False,b_fleig=True)
        cov_gauss = GaussCov(X.clone(),n,d,rho,b_fleig=True)
        cov_sep = SeparateCov(X.clone(),n,d,rho,b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(),args)

        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)

        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_gauss_paths.append(torch.norm(cov-cov_gauss,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        err_cpt1_paths.append(torch.norm(cov-cov_cpt1,'fro'))
        err_cpt2_paths.append(torch.norm(cov-cov_cpt2,'fro'))
        err_cpt3_paths.append(torch.norm(cov-cov_cpt3,'fro'))
        err_cpt4_paths.append(torch.norm(cov-cov_cpt4,'fro'))
        err_cpt5_paths.append(torch.norm(cov-cov_cpt5,'fro'))
    write_output(ns,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(ns,err_gauss_paths,strfolder,'err_gauss_paths.txt')
    write_output(ns,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(ns,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(ns,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_output(ns,err_cpt1_paths,strfolder,'err_cpt1_paths.txt')
    write_output(ns,err_cpt2_paths,strfolder,'err_cpt2_paths.txt')
    write_output(ns,err_cpt3_paths,strfolder,'err_cpt3_paths.txt')
    write_output(ns,err_cpt4_paths,strfolder,'err_cpt4_paths.txt')
    write_output(ns,err_cpt5_paths,strfolder,'err_cpt5_paths.txt')
    write_text(params,strfolder,'params.txt')       


def test_d(args,ds,strfolder,params,seeds):
    n = args.n
    s = args.s
    N = args.N
    rho = args.total_budget
        
    Ps1 = [args.total_budget]
    Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]

    err_em_paths = []
    err_gauss_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    err_cpt1_paths = []
    err_cpt2_paths = []
    err_cpt3_paths = []
    err_cpt4_paths = []
    err_cpt5_paths = []
    
    for j in range(len(ds)):
        d = int(ds[j])
        args.d = d
        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ', float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=False,b_fleig=True)
        cov_gauss = GaussCov(X.clone(),n,d,rho,b_fleig=True)
        cov_sep = SeparateCov(X.clone(),n,d,rho,b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(),args)
        
        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)

        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_gauss_paths.append(torch.norm(cov-cov_gauss,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        err_cpt1_paths.append(torch.norm(cov-cov_cpt1,'fro'))
        err_cpt2_paths.append(torch.norm(cov-cov_cpt2,'fro'))
        err_cpt3_paths.append(torch.norm(cov-cov_cpt3,'fro'))
        err_cpt4_paths.append(torch.norm(cov-cov_cpt4,'fro'))
        err_cpt5_paths.append(torch.norm(cov-cov_cpt5,'fro'))
    write_output(ds,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(ds,err_gauss_paths,strfolder,'err_gauss_paths.txt')
    write_output(ds,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(ds,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(ds,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_output(ds,err_cpt1_paths,strfolder,'err_cpt1_paths.txt')
    write_output(ds,err_cpt2_paths,strfolder,'err_cpt2_paths.txt')
    write_output(ds,err_cpt3_paths,strfolder,'err_cpt3_paths.txt')
    write_output(ds,err_cpt4_paths,strfolder,'err_cpt4_paths.txt')
    write_output(ds,err_cpt5_paths,strfolder,'err_cpt5_paths.txt')
    write_text(params,strfolder,'params.txt')                      
            

def test_rho(args,rhos,strfolder,params,seeds):
    d = args.d
    n = args.n
    s = args.s
    N = args.N
            
    err_em_paths = []
    err_gauss_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    err_cpt1_paths = []
    err_cpt2_paths = []
    err_cpt3_paths = []
    err_cpt4_paths = []
    err_cpt5_paths = []

    for j in range(len(rhos)):
        rho = rhos[j]
        args.total_budget = rho
        Ps1 = [args.total_budget]
        Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]

        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ',float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=False,b_fleig=True)
        cov_gauss = GaussCov(X.clone(),n,d,rho,b_fleig=True)
        cov_sep = SeparateCov(X.clone(),n,d,rho,b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(),args)
        
        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_gauss_paths.append(torch.norm(cov-cov_gauss,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        err_cpt1_paths.append(torch.norm(cov-cov_cpt1,'fro'))
        err_cpt2_paths.append(torch.norm(cov-cov_cpt2,'fro'))
        err_cpt3_paths.append(torch.norm(cov-cov_cpt3,'fro'))
        err_cpt4_paths.append(torch.norm(cov-cov_cpt4,'fro'))
        err_cpt5_paths.append(torch.norm(cov-cov_cpt5,'fro'))  
    write_output(rhos,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(rhos,err_gauss_paths,strfolder,'err_gauss_paths.txt')
    write_output(rhos,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(rhos,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(rhos,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_output(rhos,err_cpt1_paths,strfolder,'err_cpt1_paths.txt')
    write_output(rhos,err_cpt2_paths,strfolder,'err_cpt2_paths.txt')
    write_output(rhos,err_cpt3_paths,strfolder,'err_cpt3_paths.txt')
    write_output(rhos,err_cpt4_paths,strfolder,'err_cpt4_paths.txt')
    write_output(rhos,err_cpt5_paths,strfolder,'err_cpt5_paths.txt')
    write_text(params,strfolder,'params.txt') 
    
    
def test_Ns(args,Ns,strfolder,params,seeds):
    n = args.n
    d = args.d
    s = args.s
    rho = args.total_budget
    
    Ps1 = [args.total_budget]
    Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]

    err_em_paths = []
    err_gauss_paths = []
    err_sep_paths = []
    err_adapt_paths = []
    err_zero_paths = []
    err_cpt1_paths = []
    err_cpt2_paths = []
    err_cpt3_paths = []
    err_cpt4_paths = []
    err_cpt5_paths = []
    
    for j in  range(len(Ns)):
        N = int(Ns[j])
        X = gen_synthetic_data_fix(d,int(n),s,N,seed=seeds[j])
        cov = torch.mm(X.t(),X)/n
        print('trace: ', float(torch.trace(cov)))
        cov_em = EMCov(X.clone(),args,b_budget=False,b_fleig=True)
        cov_gauss = GaussCov(X.clone(),n,d,rho,b_fleig=True)
        cov_sep = SeparateCov(X.clone(),n,d,rho,b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(),args)
        
        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)
        err_em_paths.append(torch.norm(cov-cov_em,'fro'))
        err_gauss_paths.append(torch.norm(cov-cov_gauss,'fro'))
        err_sep_paths.append(torch.norm(cov-cov_sep,'fro'))
        err_adapt_paths.append(torch.norm(cov-cov_adapt,'fro'))
        err_zero_paths.append(torch.norm(cov,'fro'))
        err_cpt1_paths.append(torch.norm(cov-cov_cpt1,'fro'))
        err_cpt2_paths.append(torch.norm(cov-cov_cpt2,'fro'))
        err_cpt3_paths.append(torch.norm(cov-cov_cpt3,'fro'))
        err_cpt4_paths.append(torch.norm(cov-cov_cpt4,'fro'))
        err_cpt5_paths.append(torch.norm(cov-cov_cpt5,'fro'))
        
    write_output(Ns,err_em_paths,strfolder,'err_em_paths.txt')
    write_output(Ns,err_gauss_paths,strfolder,'err_gauss_paths.txt')
    write_output(Ns,err_sep_paths,strfolder,'err_sep_paths.txt')
    write_output(Ns,err_adapt_paths,strfolder,'err_adapt_paths.txt')
    write_output(Ns,err_zero_paths,strfolder,'err_zero_paths.txt')
    write_output(Ns,err_cpt1_paths,strfolder,'err_cpt1_paths.txt')
    write_output(Ns,err_cpt2_paths,strfolder,'err_cpt2_paths.txt')
    write_output(Ns,err_cpt3_paths,strfolder,'err_cpt3_paths.txt')
    write_output(Ns,err_cpt4_paths,strfolder,'err_cpt4_paths.txt')
    write_output(Ns,err_cpt5_paths,strfolder,'err_cpt5_paths.txt')
    write_text(params,strfolder,'params.txt')  
    
def make_summary(strfolder,names,num=6):
    dict_results = {}
    dict_headers = {}
    dirlist = glob(strfolder+'/test_run*')
    n = len(dirlist)
    for name in names:
        dict_results[name] = np.zeros((num,n))
    for i in range(n):
        dir = dirlist[i]
        for name in names:
            filename = name+'_paths.txt'
            data = np.genfromtxt(dir+'/'+filename)
            dict_results[name][:,i] = data[:,1]
            if not(name in dict_headers.keys()):
                dict_headers[name] = data[:,0]
    folderout = strfolder+'/summary/'
    if not os.path.isdir(folderout):
        os.makedirs(folderout)
    for name in names:
        filename = name+'_summary.txt'
        x = dict_headers[name]
        y = np.mean(dict_results[name],axis=1)
        np.savetxt(folderout+'/'+filename,np.transpose([x,y]))
        
