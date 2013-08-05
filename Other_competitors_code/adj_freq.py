## Perform OneHot encoding, creating a "rare event" vector for samples which 
## occured less than a set 'min_freq' parameter. if max_dim is set, the return
## sparse matrix won't have more than 'max_dim' columns
def OneHot_freq(data, min_freq=1, max_dim=None, keymap=None):
    data, keymap = OneHotEncoder(data, keymap=keymap); rare_vector = np.zeros((data.shape[0],1))
    if min_freq > 1:
        frequent = (data.sum(axis=0) >= min_freq).nonzero()[1] 
        frequent = frequent.__array__().reshape(frequent.shape[1],)
        while max_dim is not None and len(frequent) > max_dim:
            # increase the min_freq to reduce the numbers of dimensions
            min_freq = min_freq+1 
            frequent = (data.sum(axis=0) >= min_freq).nonzero()[1] 
            frequent = frequent.__array__().reshape(frequent.shape[1],)

        not_frequent = (data.sum(axis=0) < min_freq).nonzero()[1] # frequent columns idx
        not_frequent = not_frequent.__array__().reshape(not_frequent.shape[1]) # not frequent columns idx
        
        rare_vector = data[:,not_frequent].sum(axis=1) # set rare vector to be everything below min_freq
        if rare_vector.sum() > 0: #something is rare
            return sparse.hstack([rare_vector, data[:,frequent]]).tocsr(), keymap
    return data, keymap


if __name__ == '__main__':

    ## Will store the OneHot representation for either test & training set
    ## into two list (Xts & Xts_test). Including their relative keymaps
    num_features = X_train_all.shape[1]; frequencies = []
    for i in range(num_features):
        # set of frequencies to 1 for all features
        frequencies.append(1)
        
    print "Encoding data to OneHot rep..",; init = time.time()
    Xts = []; Xts_test = []; keymaps = [];
    for xt, keymap in [OneHot_freq(np.vstack((X_train_all[:,[i]], X_test_all[:,[i]]))
                                   , min_freq = frequencies[i]
                                   , max_dim = None) for i in range(num_features)]:
        Xts.append(xt[:num_train,:]); keymaps.append(keymap);
        Xts_test.append(xt[num_train:,:])
    print ".. done! Time : %.2fm" % ((time.time()-init)/60.0)
    
    ... end of the greedy selection loop ...

    ### ADJUST FREQUENCIES
    ## Basically, given the initial baseline cross validation score, it will process
    ## each "good features" increasing the minimum frequencies by 1 each time. If an
    ## improvement OR equal score is found, it will keep increasing the minimum
    ## frequency for that feature. The reason why will keep increasing on equal score
    ## is to handle high frequencies features, and eventually reduce the dimensionality
    ## to address eventual overfitting.
    ##
    ## While there is an improvement, it will keep looping until no further improvements.
    ## You can eventually re-run the greey feature selection afterword to look for other
    ## useful features to add to your new "good feature"
    N = 10; had_improvement = True
    while had_improvement:
        had_improvement = False
        for i in good_features: # RESET TO OPTIMAL VALUES
            Xts[i] = OneHot_freq(np.vstack((X_train_all[:,[i]], X_test_all[:,[i]]))
                    , min_freq = frequencies[i], max_dim = None)[0][:num_train,:]
            
        Xt = sparse.hstack([Xts[k] for k in good_features]).tocsr()
        best_auc = cv_loop(Xt,y,model,N)
        print "Baseline: %f" % (best_auc)
        for j in sorted(good_features, reverse=True):
        # try higher order first | de gustibus!
            Xt = sparse.hstack([Xts[k] for k in good_features]).tocsr()
            print "Try modifing frequency for fea #%i | Current : %i | AUC: %f" % (j,
                               frequencies[j], best_auc)
            increase = 0;
            while True:
                increase += 1
                print "    trying increasing by %i ..." % (increase),
                Xts[j] = OneHot_freq(np.vstack((X_train_all[:,[j]], X_test_all[:,[j]]))
                    , min_freq = frequencies[j]+increase, max_dim = None)[0][:num_train,:]
                Xt = sparse.hstack([Xts[k] for k in good_features]).tocsr()
                cur_auc = cv_loop(Xt,y,model,N)[0]
                print "AUC %f" % (cur_auc),
                if cur_auc >= best_auc:
                # equal in case reducing the frequency won't affect
                    best_auc = cur_auc; print " | continue..."
                else:
                    print " | exit!"; break
                
            if increase > 1:
                frequencies[j] += (increase-1) # currect frequency (minus last increase)
                print "New minimum frequency for fea #%i is %i" % (j, frequencies[j])
                had_improvement = True

            ## RESET FEA to NEW optimal value
            Xts[j] = OneHot_freq(np.vstack((X_train_all[:,[j]], X_test_all[:,[j]]))
                    , min_freq = frequencies[j], max_dim = None)[0][:num_train,:]    


