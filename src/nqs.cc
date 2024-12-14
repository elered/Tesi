/*
############################ COPYRIGHT NOTICE ##################################

Code provided by G. Carleo and M. Troyer, written by G. Carleo, December 2016.

Permission is granted for anyone to copy, use, modify, or distribute the
accompanying programs and documents for any purpose, provided this copyright
notice is retained and prominently displayed, along with a complete citation of
the published version of the paper:
 ______________________________________________________________________________
| G. Carleo, and M. Troyer                                                     |
| Solving the quantum many-body problem with artificial neural-networks        |
|______________________________________________________________________________|

The programs and documents are distributed without any warranty, express or
implied.

These programs were written for research purposes only, and are meant to
demonstrate and reproduce the main results obtained in the paper.

All use of these programs is entirely at the user's own risk.

################################################################################
*/


#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <complex>
#include <fstream>
#include <cassert>
#include <random>
#include <ctime>

class Nqs{

  //Neural-network weights (sono complessi)
  std::vector<std::vector<std::complex<double> > > W_;

  //Neural-network visible bias
  std::vector<std::complex<double> > a_;
  std::vector<std::complex<double> > acopy_;

  //Neural-network hidden bias
  std::vector<std::complex<double> > b_;

  //Number of hidden units
  int nh_;

  //Number of visible units
  int nv_;

  //Total number of parameters
  int npar_;

  //per la generazione dei parametri iniziali
  double sigma_;

  //look-up tables
  std::vector<std::complex<double> > Lt_;

  //Useful quantities for safe computation of ln(cosh(x))
  const double log2_;

public:

  Nqs(std::string filename):log2_(std::log(2.)){
    //LoadParametersFromFile(filename);
    LoadParameters(filename);
    npar_ = (2*nv_)+nh_+(2*nv_)*nh_;
  }

  //computes the logarithm of the wave-function; (calcola i valori dell'rbm)
  inline std::complex<double> LogVal(const std::vector<int> & state)const{

    std::complex<double> rbm(0.,0.); //ho due valori perchè sono parte reale e immaginaria

    for(int v=0;v<2*nv_;v++){
      rbm+=a_[v]*double(state[v]);
    }

    for(int h=0;h<nh_;h++){

      std::complex<double> thetah=b_[h];

      for(int v=0;v<2*nv_;v++){
        thetah+=double(state[v])*(W_[v][h]);
      }

      rbm+=Nqs::lncosh(thetah);
    }

    return rbm;
  }

  //computes the logarithm of Psi(state')/Psi(state)
  //where state' is a state with a certain number of flipped spins
  //the vector "flips" contains the sites to be flipped
  //look-up tables are used to speed-up the calculation
  inline std::complex<double> LogPoP(const std::vector<int> & state,const std::vector<int> & flips)const{ //prende in input il vettore con lo stato e il vettore con gli indici da flippare

    if(flips.size()==0){ //caso in cui non flippo nessuno spin
      return 0.;
    }

    std::complex<double> logpop(0.,0.);

    //Change due to the visible bias
    for(const auto & flip : flips){ //ciclo for che itera attraverso ogni elemento del vettore flips
      logpop-=a_[flip]*2.*double(state[flip]);
    }

    //Change due to the interaction weights
    for(int h=0;h<nh_;h++){
      const std::complex<double> thetah=Lt_[h];
      std::complex<double> thetahp=thetah;

      for(const auto & flip : flips){
        thetahp-=2.*double(state[flip])*(W_[flip][h]);
      }
      logpop+= ( Nqs::lncosh(thetahp)-Nqs::lncosh(thetah) );
    }

    return logpop;
  }

  //calcola il valore della probabilità della variazione di configurazione nella rete di Boltzmann
  inline std::complex<double> PoP(const std::vector<int> & state,const std::vector<int> & flips)const{
    return std::exp(LogPoP(state,flips));
  }

  //initialization of the look-up tables (sata per calcolare velocemente le variazioni nei pesi durante le iterazioni)
  void InitLt(const std::vector<int> & state){
    Lt_.resize(nh_);

    for(int h=0;h<nh_;h++){
      Lt_[h]=b_[h];             //Inizializza il valore di `Lt_[h]` con il bias corrispondente
      for(int v=0;v<2*nv_;v++){
        Lt_[h]+=double(state[v])*(W_[v][h]);  //Aggiorna `Lt_[h]` con i contributi dai pesi visibili
      }
    }

  }

  //updates the look-up tables after spin flips
  //the vector "flips" contains the indices of sites to be flipped
  void UpdateLt(const std::vector<int> & state,const std::vector<int> & flips){
    if(flips.size()==0){
      return;  //se non c'è nessuno spin flip non faccio nulla
    }

    for(int h=0;h<nh_;h++){ //per ogni neurone nascosto
      for(const auto & flip : flips){ //per ogni spin flippato
        Lt_[h]-=2.*double(state[flip])*W_[flip][h]; //aggiorna Lt_[h]
      }
    }
  }


  //loads the parameters of the wave-function from a given file -> now it generates initial parameters too
  void LoadParameters(std::string filename){

    std::ifstream fin(filename.c_str());

    if(!fin.good()){
      std::cerr<<"# Error : Cannot load from file "<<filename<<" : file not found."<<std::endl;
      std::abort();
    }

    fin>>nv_;
    fin>>nh_;
    fin>>sigma_;

    if(!fin.good() || nv_<0 || nh_<0){
      std::cerr<<"# Trying to load from an invalid file.";
      std::cerr<<std::endl;
      std::abort();
    }

    if(!fin.good()){
      std::cerr<<"# Trying to load from an invalid file.";
      std::cerr<<std::endl;
      std::abort();
    }

    a_.resize(2*nv_);
    acopy_.resize(nv_);
    b_.resize(nh_);
    W_.resize(2*nv_,std::vector<std::complex<double> > (nh_));

      std::mt19937 gen_;
      gen_.seed(std::time(nullptr));
      std::normal_distribution<double> distribution(0,sigma_);

    for(int i=0;i<2*nv_;i++){
      a_[i]= std::complex<double>(distribution(gen_),distribution(gen_));
    }
    //a_=acopy_;
    //a_.insert(a_.end(), acopy_.begin(), acopy_.end());

    for(int j=0;j<nh_;j++){
      b_[j]= std::complex<double>(distribution(gen_),distribution(gen_));
    }

    for(int i=0;i<2*nv_;i++){
      for(int j=0;j<nh_;j++){
        W_[i][j]= std::complex<double>(distribution(gen_),distribution(gen_));
      }
    }
    
    std::cout<<"# NQS w/ Spin loaded with random values "<<filename<<std::endl;
    std::cout<<"# N_visible = "<<nv_<<"(x2)  N_hidden = "<<nh_<<std::endl;
  }

  //loads the parameters of the wave-function from a given file
  void LoadParametersFromFile(std::string filename){

    std::ifstream fin(filename.c_str());

    if(!fin.good()){
      std::cerr<<"# Error : Cannot load from file "<<filename<<" : file not found."<<std::endl;
      std::abort();
    }

    fin>>nv_;
    fin>>nh_;

    if(!fin.good() || nv_<0 || nh_<0){
      std::cerr<<"# Trying to load from an invalid file.";
      std::cerr<<std::endl;
      std::abort();
    }

    a_.resize(2*nv_);
    acopy_.resize(nv_);
    b_.resize(nh_);
    W_.resize(2*nv_,std::vector<std::complex<double> > (nh_));

    for(int i=0;i<nv_;i++){
      fin>>acopy_[i];
    }
    a_=acopy_;
    a_.insert(a_.end(), acopy_.begin(), acopy_.end());

    for(int j=0;j<nh_;j++){
      fin>>b_[j];
    }

    for(int i=0;i<2*nv_;i++){
      for(int j=0;j<nh_;j++){
        fin>>W_[i][j];
      }
    }
    
    if(!fin.good()){
      std::cerr<<"# Trying to load from an invalid file.";
      std::cerr<<std::endl;
      std::abort();
    }

    std::cout<<"# NQS w/ Spin loaded from file "<<filename<<std::endl;
    std::cout<<"# N_visible = "<<nv_<<"(x2)  N_hidden = "<<nh_<<std::endl;
  }

  //used for variational MC:
  std::vector<std::complex<double>> GetParameters(){

     std::vector<std::complex<double> > pars; //vettore di numeri complessi
     pars.resize(npar_);
     int k=0;
     for(;k<2*nv_;k++){
       pars[k]=a_[k];
     }

     for(int p=0;p<nh_;p++){ //continuo a rimepire il vettore pars dall'indice k
         pars[k]=b_[p];
         k++;
     }

     for(int i=0;i<2*nv_;i++){
       for(int j=0;j<nh_;j++){
         pars[k]=W_[i][j];
         k++;
       }
     }

     return pars;
  }

  void SetParameters(const std::vector<std::complex<double>> & pars){

        int k=0;
        for(;k<2*nv_;k++){
          a_[k]=pars[k];
        }
        for(int p=0;p<nh_;p++){
            b_[p]=pars[k];
            k++;
        }
        for(int i=0;i<2*nv_;i++){
          for(int j=0;j<nh_;j++){
            W_[i][j]=pars[k];
            k++;
          }
        }
  }


  //deriva il logaritmo della funzione d'onda rispetto a tutti i parametri
  std::vector<std::complex<double>> DerLog(const std::vector<int> & state){ 
    std::vector<std::complex<double>> der;
    der.resize(npar_);

    int k = 0;
    for(;k<2*nv_;k++){
      der[k]=state[k];
    }

    std::vector<std::complex<double>> tanhThetas;
    tanhThetas.resize(nh_);
    tanh(Lt_,tanhThetas); //calcola la tangente iperbolica per ogni elemento di Lt_ e salva i risultati in tanhThetas
    for(int p=0;p<nh_;p++){
      der[k]=tanhThetas[p];
      k++;
    }

    for(int i=0;i<2*nv_;i++){
      for(int j=0;j<nh_;j++){
        der[k]=tanhThetas[j]*double(state[i]);
        k++;
      }
    }
    return der;
  }

  //tanh for complex vectors
  void tanh(const std::vector<std::complex<double>> & x,std::vector<std::complex<double>>& y){
    for(int i=0;i<x.size();i++){
      y[i]=std::tanh(x[i]);
    }
  }

  //ln(cos(x)) for real argument
  //for large values of x we use the asymptotic expansion
  inline double lncosh(double x)const{
    const double xp=std::abs(x);
    if(xp<=12.){ 
      return std::log(std::cosh(xp)); //Se xp è relativamente piccolo (<= 12), calcola log(cosh(x))
    }
    else{
      return xp-log2_; // Se xp è grande (> 12), utilizza un'approssimazione stabile
    }
  }

  //ln(cos(x)) for complex argument
  //the modulus is computed by means of the previously defined function
  //for real argument
  inline std::complex<double> lncosh(std::complex<double> x)const{
    const double xr=x.real();
    const double xi=x.imag();

    std::complex<double> res=Nqs::lncosh(xr);
    res +=std::log( std::complex<double>(std::cos(xi),std::tanh(xr)*std::sin(xi)) );

    return res;
  }

  //total number of spins
  //equal to the number of visible units
  inline int Nspins()const{
    return nv_;
  }

  inline int NHspins()const{
    return nh_;
  }

  int GetNpar()const{
    return npar_;
  }
};
