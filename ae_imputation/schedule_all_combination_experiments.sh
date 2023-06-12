
spatial=$1
hyper=$2
iterations=$3

# if $iterations is empty set iterations to 1
if [ "$iterations" == "" ]; then
  iterations=1
fi

echo "spatial=${spatial}" "hyper=${hyper}" "iterations=${iterations}"

if [ "$hyper" == "" ]; then
  ./ae_imputation/schedule_experiments.sh ip zero "" "${spatial}" $iterations
  ./ae_imputation/schedule_experiments.sh ip mean "" "${spatial}" $iterations
  ./ae_imputation/schedule_experiments.sh ip zero "1" "${spatial}" $iterations
  ./ae_imputation/schedule_experiments.sh ip mean "1" "${spatial}" $iterations

  ./ae_imputation/schedule_experiments.sh exp zero "" "${spatial}" $iterations
  ./ae_imputation/schedule_experiments.sh exp mean "" "${spatial}" $iterations
  ./ae_imputation/schedule_experiments.sh exp zero "1" "${spatial}" $iterations
  ./ae_imputation/schedule_experiments.sh exp mean "1" "${spatial}" $iterations
else
  echo "Hyper parameter tuning set"
  #./ae_imputation/schedule_experiments.sh ip zero "" "${spatial}" 10 1
  ./ae_imputation/schedule_experiments.sh ip mean "" "${spatial}" $iterations 1
  #./ae_imputation/schedule_experiments.sh ip zero "1" "${spatial}" 10 1
  #./ae_imputation/schedule_experiments.sh ip mean "1" "${spatial}" 10 1

  #./ae_imputation/schedule_experiments.sh exp zero "" "${spatial}" 10 1
  ./ae_imputation/schedule_experiments.sh exp mean "" "${spatial}" $iterations 1
  #./ae_imputation/schedule_experiments.sh exp zero "1" "${spatial}" 10 1
  #./ae_imputation/schedule_experiments.sh exp mean "1" "${spatial}" 10 1

fi