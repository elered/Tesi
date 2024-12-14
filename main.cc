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

#include "src/nqs_paper.hh"


int main(int argc, char *argv[]){

  auto opts=ReadOptions(argc,argv);

  //Definining the neural-network wave-function
  Nqs wavef(opts["filename"]);

  int nsweeps=std::stod(opts["nsweeps"]); //stod converte stringa in double
  int nspins=wavef.Nspins();

  //Problem hamiltonian inferred from file name
  std::string model=opts["model"];

  bool printastes=opts.count("filestates");  //true se c'è l'opzione e poi attiva la procedura nel sampler

  int seed=std::stoi(opts["seed"]); //stoi converte stringa in int


  //erano quelli che c'erano nel codice iniziale di carleo
  /*if(model=="Ising1d"){
    double hfield=std::stod(opts["hfield"]);
    Ising1d hamiltonian(nspins,hfield);

    //Defining and running the sampler
    Sampler<Nqs,Ising1d> sampler(wavef,hamiltonian,seed);
    if(printastes){
      sampler.SetFileStates(opts["filestates"]);
    }
    sampler.Run(nsweeps);
  }
  else if(model=="Heisenberg1d"){
    double jz=std::stod(opts["jz"]);
    Heisenberg1d hamiltonian(nspins,jz);

    //Defining and running the sampler
    Sampler<Nqs,Heisenberg1d> sampler(wavef,hamiltonian,seed);
    if(printastes){
      sampler.SetFileStates(opts["filestates"]);
    }
    sampler.Run(nsweeps);
  }
  else if(model=="Heisenberg2d"){
    double jz=std::stod(opts["jz"]);
    Heisenberg2d hamiltonian(nspins,jz);

    //Defining and running the sampler
    Sampler<Nqs,Heisenberg2d> sampler(wavef,hamiltonian,seed);
    if(printastes){
      sampler.SetFileStates(opts["filestates"]);
    }
    sampler.Run(nsweeps);
  }*/
  if(model=="Trap3d"){
    double magnetization=std::stod(opts["M"]);
	std::cout<<"magnetization is: "<<magnetization<<std::endl;

    int Nup=0;
    int Ndown=0;
    if(magnetization>=1){
      std::cout<<"How many spins up?"<<std::endl;
      std::cin>>Nup;
      std::cout<<"How many spins down?"<<std::endl;
      std::cin>>Ndown;

      if( (Ndown+Nup) !=magnetization ){
        std::cout<<"Error. The total number of spins must be equal to the magnetization."<<std::endl;
        return -1;
      }
    }
    
    Trap3d hamiltonian(nspins, magnetization);

    //Defining and running the sampler
    Sampler<Nqs,Trap3d> sampler(wavef,hamiltonian,seed, Nup, Ndown);
    if(printastes){
      sampler.SetFileStates(opts["filestates"]);
    }
    sampler.Run(nsweeps);
  }
  if(model=="Coulomb3d"){
    double magnetization=std::stod(opts["M"]);
	std::cout<<"magnetization is: "<<magnetization<<std::endl;

    int Nup=0;
    int Ndown=0;
    if(magnetization>=1){
      std::cout<<"How many spins up?"<<std::endl;
      std::cin>>Nup;
      std::cout<<"How many spins down?"<<std::endl;
      std::cin>>Ndown;

      if( (Ndown+Nup) !=magnetization ){
        std::cout<<"Error. The total number of spins must be equal to the magnetization."<<std::endl;
        return -1;
      }
    }
    
    Coulomb3d hamiltonian(nspins, magnetization);

    //Defining and running the sampler
    Sampler<Nqs,Coulomb3d> sampler(wavef,hamiltonian,seed, Nup, Ndown);
    if(printastes){
      sampler.SetFileStates(opts["filestates"]);
    }
    sampler.Run(nsweeps);
  }
  else if(model=="Coulomb1d"){
    double magnetization=std::stod(opts["M"]);
	std::cout<<"magnetization is: "<<magnetization<<std::endl;

    int Nup=0;
    int Ndown=0;
    if(magnetization>=1){
      std::cout<<"How many spins up?"<<std::endl;
      std::cin>>Nup;
      std::cout<<"How many spins down?"<<std::endl;
      std::cin>>Ndown;

      if( (Ndown+Nup) !=magnetization ){
        std::cout<<"Error. The total number of spins must be equal to the magnetization."<<std::endl;
        return -1;
      }
    }
    
    Coulomb1d hamiltonian(nspins, magnetization);

    //Defining and running the sampler
    Sampler<Nqs,Coulomb1d> sampler(wavef,hamiltonian,seed, Nup, Ndown);
    if(printastes){
      sampler.SetFileStates(opts["filestates"]);
    }
    sampler.Run(nsweeps);
  }
  else if(model=="Trap1d"){
    double magnetization=std::stod(opts["M"]);
	  std::cout<<"magnetization is: "<<magnetization<<std::endl;

    int Nup=0;
    int Ndown=0;
    if(magnetization>=1){
      std::cout<<"How many spins up?"<<std::endl;
      std::cin>>Nup;
      std::cout<<"How many spins down?"<<std::endl;
      std::cin>>Ndown;

      if( (Ndown+Nup) !=magnetization ){
        std::cout<<"Error. The total number of spins must be equal to the magnetization."<<std::endl;
        return -1;
      }
    }

    Trap1d hamiltonian(nspins, magnetization);

    //Defining and running the sampler
    Sampler<Nqs,Trap1d> sampler(wavef,hamiltonian,seed, Nup, Ndown);
    if(printastes){
      sampler.SetFileStates(opts["filestates"]);
    }
    sampler.Run(nsweeps);
  }
  else{
    std::cerr<<"#The given input file does not correspond to one of the implemented problem hamiltonians";
    std::abort();
  }

}
