from featureGenerator import extractFeatures


def main():
    filename = "verizonData.csv"
    feature_object = extractFeatures.ExtractFeature(filename)
    feature_object.openfile()
    feature_object.gen_feature()
    feature_object.naive_train()
    feature_object.svm_train()


main()
