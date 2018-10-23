# covtype
cd covtype
if [ ! -f covtype-test-1.csv ]; then
   echo "Downloading covtype test..."
   wget https://sid.erda.dk/share_redirect/bx3kbiD08L/covtype-test-1.csv
else
   echo "skipping covtype test, file exists..."
fi
if [ ! -f covtype-train-1.csv ]; then
   echo "Downloading covtype train..."
   wget https://sid.erda.dk/share_redirect/bx3kbiD08L/covtype-train-1.csv
else
   echo "Skipping covtype train, file exists..."
fi
cd ..

# susy
cd susy
if [ ! -f SUSY.csv ]; then
   echo "downloading and unzipping susy.."
   wget http://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
   gunzip SUZY.csv.gz
else
   echo "Skipping suzy, file exists..."
fi   
cd ..

# higgs
cd higgs
if [ ! -f HIGGS.csv ]; then
   echo "Downloading and unzipping higgs..."
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
   gunzip HIGGS.csv.gz
else
   echo "Skipping higgs, file exists..."
fi
cd ..



