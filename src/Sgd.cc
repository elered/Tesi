#include <iostream>
//#include <Eigen/Core>
//#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>

class Sgd{

  //decay constant, learning parameter
  std::complex<double> eta_; //learning rate

  int npar_; //numero di parametri

  double l2reg_; //coefficiente di regolarizzazione L2, che penalizza i pesi grandi per prevenire overfitting

  double dropout_p_; //Probabilità di dropout, utilizzata per regolare la rete disabilitando alcuni neuroni durante l'allenamento

  double momentum_; //termine di momento, che accelera il training spingendo la direzione del gradiente nella direzione delle iterazioni precedenti

  std::mt19937 rgen_;

  double decay_factor_; // Fattore di decadimento per il learning rate, usato per ridurre gradualmente eta_ durante l'allenamento

  int niter_;

  double eta1_;

  bool etaDecay_; // Flag per abilitare/disabilitare il decadimento del learning rate


public:

  Sgd(std::complex<double> eta,double eta1, bool etaDecay=true, double l2reg=0,double momentum=0,double dropout_p=0):
    eta_(eta),eta1_(eta1),etaDecay_(etaDecay),l2reg_(l2reg),dropout_p_(dropout_p),momentum_(momentum){
    npar_=-1; // per segnalare che il numero di parametri non è ancora stato definito
    decay_factor_=1;
    niter_=0;
  }

  void SetNpar(int npar){
    npar_=npar;
  }

  void Update(const std::vector<std::complex<double>> & grad, std::vector<std::complex<double>> & pars){ //pars è il vettore dei parametri
    assert(npar_>0); // verifica che il numero di parametri sia maggiore di zero
    if(etaDecay_==true){
      eta_=eta_*(1/std::sqrt(1+niter_/eta1_)); //riduco il learning rate a ogni iterazione
    }
    //eta_*=decay_factor_;

    //double gradnorm=1./sqrt(grad.norm());
    //per ora non normalizzo
    std::uniform_real_distribution<double> distribution(0,1); //cmq non la uso

    for(int i=0;i<npar_;i++){
      if(distribution(rgen_)>dropout_p_){
        pars[i]=(1.-momentum_)*pars[i] - (grad[i]+l2reg_*pars[i]) * eta_; //*gradnorm
      }
    }


    niter_+=1;
  }

  //void SetDecayFactor(double decay_factor){
  //  assert(decay_factor<1);
  //  decay_factor_=decay_factor;
  //}

  //void Reset(){
  //}
};
