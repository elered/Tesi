#include <iostream>
//#include <Eigen/Dense>
//#include <Eigen/IterativeLinearSolvers>
#include "eigen3/Eigen/Eigenvalues"
#include <random>
#include <complex> //libreria che mi permette di lavorare coi numeri complessi
#include <vector>
#include <fstream>

using namespace std;
using namespace Eigen; //eigen è una libreria per l'algebra lineare

template<class Hamiltonian,class Nqs,class Sampler,class Optimizer> class Variational{

  Hamiltonian & ham_;
  Nqs & nqs_;
  Sampler & sampler_;
  Optimizer opt_;

  int npar_; //numero parametri del sistema
  vector<vector<int> > configSamp_;
  vector<vector<int> > indexSamp_;
  vector<vector<complex<double>>> derLogs_; //vettore che contiene le derivate logaritmiche calcolate dal sampler
  vector<double> elocs_; //energie locali
  double elocmean_; //media energie locali
  double sigmaElocmean_; //sigma energia locale media
  double loss_;
  int nsweeps_; //numero totale campionamenti
  vector<complex<double>> grad_; //gradiente
  vector<vector<complex<double>>> QFisher0_; //matrice di Fisher
  //MatrixXcd QFisher0_;
  //complex<double>* QFisher0_; per lapack-> costruttore, distruttore, calcQFisher e UpdateParametersSR e include di SR

  ofstream out_;

  int Iter0_; //tengo conto del numero di iterazioni(?)

public:

  Variational(Sampler & sampler,Optimizer & opt):ham_(sampler.GetHamiltonian()),nqs_(sampler.GetNqs()),sampler_(sampler),opt_(opt){

    npar_=nqs_.GetNpar();

    grad_.resize(npar_);
    opt_.SetNpar(npar_);
    Iter0_=0;
    //QFisher0_ = new complex<double>[npar_*npar_];

  }

  ~Variational(){
    //delete[] QFisher0_;
    out_.close();
  }

  void Sample(){
    sampler_.Sample(nsweeps_); //chiama un metodo sample nel Sampler per eseguire il campionamento
    configSamp_= sampler_.GetSamplingConfig(); //riempio con le configurazioni del campionamento
    indexSamp_ = sampler_.GetSamplingIndex(); //riempio con gli indici del campionamento
    elocs_.resize(nsweeps_);
    realPart(sampler_.GetSamplingEnergies(), elocs_); //riempio il vettore delle energie locali, realParte prende la parte reale
    elocmean_ = accumulate(elocs_.begin(), elocs_.end(), 0.0)/elocs_.size(); 
    derLogs_ = sampler_.GetDerLogs(); //raccoglie le derivate logaritmiche dei campionamenti
  }

  void CalcSigmaElocmean(int bins=50){ //bins è il numero di blocchi
    int blocksize=floor(double(elocs_.size())/double(bins)); //floor arrotonda all'intero più basso più vicino
    double enmean=0;
    double enmeansq=0;
    for(int i=0;i<bins;i++){
      double eblock=0;
      for(int j=i*blocksize;j<(i+1)*blocksize;j++){
        eblock+=elocs_[j];
      }
      eblock/=double(blocksize);
      double delta=eblock-enmean;
      enmean+=delta/double(i+1);
      double delta2=eblock-enmean;
      enmeansq+=delta*delta2;
    }
    enmeansq/=(double(bins-1)); //diviso n-1 ?
    // enmean; //estimated average
    sigmaElocmean_=sqrt(enmeansq/double(bins)); //estimated error
  }

  //void CalcDerLogs(){
  //  derLogs_.resize(nsweeps_,vector<complex<double>> (npar_));
  //  for(int i=0;i<nsweeps_;i++){
  //    derLogs_[i] = nqs_.DerLog(configSamp_[i]); //sulla riga cresce il numero di sweep, i sale sulla y
  //  }
  //}

  // THIS ONE USES C++ VECTORS AND EIGEN
  void CalcQuantumFisher(){
    QFisher0_.resize(npar_,vector<complex<double>> (npar_)); //ridimensiono la matrice per farla diventare quadrata nparxnpar
    vector<complex<double>> meanDerlog(npar_,(0.,0.)); //vettore che terrà la media delle derivate logaritmiche per ogni parametro
    //calcolo la media delle derivate logaritmiche per ogni parametro variazionale
    for(int i=0;i<npar_;i++){
      for(int k=0;k<nsweeps_;k++){
        meanDerlog[i] += derLogs_[k][i] / double(nsweeps_); //i è il parametro e k è il campionamento
      }
    }
    //costruzione della matrice di Fisher
    for(int i=0;i<npar_;i++){
      for(int j=0;j<npar_;j++){
        QFisher0_[i][j]= - conj(meanDerlog[j]) * meanDerlog[i];
        for(int k=0;k<nsweeps_;k++){
          QFisher0_[i][j] += conj(derLogs_[k][j])*derLogs_[k][i] / double(nsweeps_);
        }
      }
    }
  }

/*  THIS ONE USES ONLY EIGEN AND PAIRS WITH SR.UPGRADE2
void CalcQuantumFisher(){
    QFisher0_.resize(npar_,npar_);
    vector<complex<double>> meanDerlog(npar_,(0.,0.));
    for(int i=0;i<npar_;i++){
      for(int k=0;k<nsweeps_;k++){
        meanDerlog[i] += derLogs_[k][i] / double(nsweeps_);
      }
    }
    for(int i=0;i<npar_;i++){
      for(int j=0;j<npar_;j++){
        QFisher0_(i,j)= - conj(meanDerlog[i]) * meanDerlog[j];
        for(int k=0;k<nsweeps_;k++){
          QFisher0_(i,j) += conj(derLogs_[k][i])*derLogs_[k][j] / double(nsweeps_);
        }
      }
    }
  } */

    // THIS ONE USES MKL AND LAPACKE, THUS POINTERS OF COMPLEX ROW-MAJOR, AND PAIRS WITH SR.UPGRADE3
  /*  void CalcQuantumFisher(){
      vector<complex<double>> meanDerlog(npar_,(0.,0.));
      for(int i=0;i<npar_;i++){
        for(int k=0;k<nsweeps_;k++){
          meanDerlog[i] += derLogs_[k][i] / double(nsweeps_);
        }
      }
      for(int i=0;i<npar_;i++){
        for(int j=0;j<npar_;j++){
          QFisher0_[i*npar_+j] = - conj(meanDerlog[i]) * meanDerlog[j];
          for(int k=0;k<nsweeps_;k++){
            QFisher0_[i*npar_+j] += conj(derLogs_[k][i])*derLogs_[k][j] / double(nsweeps_);
          }
        }
      }
    } */

  inline complex<double> PhiTarget(vector<int> stateIndex){ //2p1d GS (rappresento questo stato tramite indici)
    int nv; //numero di siti
    nv= nqs_.Nspins();
    complex<double> phi; //ampiezza della funzione d'onda per questo stato
    phi = 1/sqrt(2) *( sin(complex<double>(M_PI*(stateIndex[0]+0.5)/nv,0)) * sin(complex<double>(M_PI*(stateIndex[1]+0.5)/nv,0)) - sin(complex<double>(M_PI*(stateIndex[0]+0.5)/nv,0)) * sin(complex<double>(M_PI*(stateIndex[1]+0.5)/nv,0)) ) ;
    return phi;
  }

  void CalcExactLossFunction(){ //exact loss function for 1particle
    int nv;
    nv= nqs_.Nspins();
    for(int i=0;i<nv;i++){
      vector<int> state(nv,0);
      state[i]=1;
      vector<int> stateIndex(1,i);
      loss_+= norm(exp(nqs_.LogVal(state))-PhiTarget(stateIndex)); //norm mi calcola il modulo quadro di un numero complesso
    }
  }

  void PretrainingGradient(){
    //CalcDerLogs();
    for (int i=0;i<npar_;i++) grad_[i]=0.;

    for(int i=0;i<nsweeps_;i++){
      complex<double> targetDistance = 1.-PhiTarget(indexSamp_[i])/ exp(nqs_.LogVal(configSamp_[i]));
      for(int j=0;j<npar_;j++){
        grad_[j] += targetDistance * conj(derLogs_[i][j]) * 2. / double(nsweeps_);
      }
    }

  }

  void TrainingGradient(){
    //CalcDerLogs();
    for(int i=0;i<npar_;i++) grad_[i]=0.;

    for(int i=0;i<nsweeps_;i++){
      complex<double> targetDistance = elocs_[i]-elocmean_;
      for(int j=0;j<npar_;j++){
        grad_[j] += targetDistance * conj(derLogs_[i][j]) * 2. / double(nsweeps_);
      }
    }
  }

  void RunPretraining(int nsweeps,int niter){
    nsweeps_=nsweeps;
    SetOutputFilePretraining();
    for(int i=0;i<niter;i++){
      Sample();

      PretrainingGradient();

      CalcSigmaElocmean();
      CalcExactLossFunction();
      PrintStats(i);

      UpdateParameters();
    }
    PrintStats(Iter0_+niter);
    Iter0_+=niter;
  }

  void RunPretrainingSR(int nsweeps,int niter){
    nsweeps_=nsweeps;
    SetOutputFilePretraining();
    for(int i=0;i<niter;i++){
      Sample();

      PretrainingGradient();
      CalcQuantumFisher();

      CalcSigmaElocmean();
      CalcExactLossFunction();
      PrintStats(i);

      UpdateParametersSR();
    }
    PrintStats(Iter0_+niter);
    Iter0_+=niter;
  }

  void RunTraining(int nsweeps,int niter){
    nsweeps_=nsweeps;
    SetOutputFileTraining("TrainingTest/Training.dat");				
    for(int i=0;i<niter;i++){
      Sample();

      TrainingGradient();

      CalcSigmaElocmean();
      CalcExactLossFunction();
      PrintStats(i);

      UpdateParameters();
    }
    PrintStats(Iter0_+niter);
    Iter0_+=niter;
  }

  void RunTrainingSR(int nsweeps,int niter, std::string outputfile){
    nsweeps_=nsweeps;
    SetOutputFileTraining(outputfile);
    for(int i=0;i<niter;i++){
      Sample();

      TrainingGradient();
      CalcQuantumFisher();

      CalcSigmaElocmean();
      CalcExactLossFunction();			//calcola distanza del mio stato dallo stato target
      PrintStats(i);

      UpdateParametersSR();
    }
    PrintStats(Iter0_+niter);
    Iter0_+=niter;
  }

  void UpdateParameters(){
    auto pars=nqs_.GetParameters();
    opt_.Update(grad_,pars); //usa lo stesso grad per training e pretraining
    nqs_.SetParameters(pars);
  }

  void UpdateParametersSR(){
    auto pars=nqs_.GetParameters();
    opt_.Update(grad_, QFisher0_, pars); //no number per vector+eigen, 1 per only eigen, 2 per lapack
    nqs_.SetParameters(pars);
  }

  void SetOutputFilePretraining(){
    out_.open("TrainingTest/Pretraining.dat", std::ios_base::out);
    if(!out_.is_open()){
        std::cerr<<"# Error : Cannot open file for writing optimization data"<<std::endl;
        std::abort();
      }
    out_<<"Niter"<<"\t"<<"Eloc"<<"\t"<<"sigma"<<"\t"<<"loss"<<"\t";
    for(int i=0;i<npar_;i++) out_<<"par"<<i<<"\t";
    out_<<endl;

    std::cout<<"# Saving sampled configuration to file "<<"TrainingTest/Pretraining.dat"<<std::endl;
  }

  void SetOutputFileTraining(std::string outputfile){
    out_.open(outputfile.c_str(), std::ios_base::out);
    if(!out_.is_open()){
        std::cerr<<"# Error : Cannot open file for writing optimization data"<<std::endl;
        std::abort();
      }
    out_<<"Niter"<<"\t"<<"Eloc"<<"\t"<<"sigma"<<"\t"<<"loss"<<"\t";
    for(int i=0;i<npar_;i++) out_<<"par"<<i<<"\t";
    out_<<endl;

    std::cout<<"# Saving sampled configuration to file "<<outputfile.c_str()<<std::endl;
  }

  void PrintStats(int i){

    out_<<i+Iter0_<<"\t"<<scientific<<elocmean_<<"\t"<<sigmaElocmean_<<"\t"<<loss_<<"\t";
    auto pars=nqs_.GetParameters();
    for(const auto& par : pars){
      out_<<"("<<par.real();
      if(par.imag()>=0.) out_<<"+";
      else out_<<"-";
      out_<<abs(par.imag())<<"j)"<<"\t";
    }
    out_<<endl;

    //cout<<i+Iter0_<<"  "<<scientific<<elocmean_<<"   "<<grad_.norm()<<" "<<rbm_.GetParameters().array().abs().maxCoeff()<<" ";
    std::cout<<i<<"th iteration: done writing optimization data"<<std::endl;
  }

  //Debug function to check that the logarithm of the derivative is
  //computed correctly
/*  void CheckDerLog(double eps=1.0e-4){
    sampler_.Reset(true);

    auto ders=rbm_.DerLog(sampler_.Visible());

    auto pars=rbm_.GetParameters();

    for(int i=0;i<npar_;i++){
      pars(i)+=eps;
      rbm_.SetParameters(pars);
      double valp=rbm_.LogVal(sampler_.Visible());

      pars(i)-=2*eps;
      rbm_.SetParameters(pars);
      double valm=rbm_.LogVal(sampler_.Visible());

      pars(i)+=eps;

      double numder=(-valm+valp)/(eps*2);

      if(std::abs(numder-ders(i))>eps*eps){
        cerr<<"Possible error on parameter "<<i<<". Expected: "<<ders(i)<<" Found: "<<numder<<endl;
      }
    }
  } */

  void realPart(const vector<complex<double>> & x,vector<double>& y){
    for(int i=0;i<x.size();i++){
      y[i]=x[i].real();
    }
  }
};
