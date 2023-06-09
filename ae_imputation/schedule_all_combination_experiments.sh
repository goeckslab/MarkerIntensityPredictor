
spatial=$1
hyper=$2

if [ "$hyper" == "" ]; then
  ./ae_imputation/schedule_experiments.sh ip zero "" "${spatial}" 10
  ./ae_imputation/schedule_experiments.sh ip mean "" "${spatial}" 10
  ./ae_imputation/schedule_experiments.sh ip zero "1" "${spatial}" 10
  ./ae_imputation/schedule_experiments.sh ip mean "1" "${spatial}" 10

  ./ae_imputation/schedule_experiments.sh exp zero "" "${spatial}" 10
  ./ae_imputation/schedule_experiments.sh exp mean "" "${spatial}" 10
  ./ae_imputation/schedule_experiments.sh exp zero "1" "${spatial}" 10
  ./ae_imputation/schedule_experiments.sh exp mean "1" "${spatial}" 10
else
  echo "Hyper parameter tuning set"
  #./ae_imputation/schedule_experiments.sh ip zero "" "${spatial}" 10 1
  ./ae_imputation/schedule_experiments.sh ip mean "" "${spatial}" 10 1
  #./ae_imputation/schedule_experiments.sh ip zero "1" "${spatial}" 10 1
  #./ae_imputation/schedule_experiments.sh ip mean "1" "${spatial}" 10 1

  #./ae_imputation/schedule_experiments.sh exp zero "" "${spatial}" 10 1
  ./ae_imputation/schedule_experiments.sh exp mean "" "${spatial}" 10 1
  #./ae_imputation/schedule_experiments.sh exp zero "1" "${spatial}" 10 1
  #./ae_imputation/schedule_experiments.sh exp mean "1" "${spatial}" 10 1

fi