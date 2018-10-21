#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH2.h"
#include "TRandom.h"
#include <vector>
void plot_root_vec()
{
   std::vector<float> *E = 0; 
   std::vector<int> *px =0; 
   std::vector<int> *py =0; 
   std::vector<int> *pz =0; 

   std::vector<float> *E_g4 = 0; 
   std::vector<int> *px_g4 =0; 
   std::vector<int> *py_g4 =0; 
   std::vector<int> *pz_g4 =0; 

  // GANs output
 //  TFile *f = new TFile("/home/svalleco/GAN/rootFiles/generationEdiscr000.root");
   //TFile *f = new TFile("/home/svalleco/rootfiles/GeneratedEnergyRoot.root");
   TFile *f = new TFile("/home/svalleco/rootfiles/generated100Rootfile.root");
   TTree *t1 = (TTree*)f->Get("ecalTree");
 
   t1->SetBranchAddress("x",&px);
   t1->SetBranchAddress("y",&py);
   t1->SetBranchAddress("z",&pz);
   t1->SetBranchAddress("E",&E);
 //Geant4 full sim
   //TFile *f_g4 = new TFile("/home/svalleco/GAN/test_nv08/testG.root");
   //TFile *f_g4 = new TFile("Geant4.root");
   //TFile *f_g4 = new TFile("/home/svalleco/rootfiles/ElectronEnergyRoot.root");
   TFile *f_g4 = new TFile("/home/svalleco/rootfiles/EG100rootfile.root");
   TTree *t1_g4 = (TTree*)f_g4->Get("ecalTree");

   t1_g4->SetBranchAddress("x",&px_g4);
   t1_g4->SetBranchAddress("y",&py_g4);
   t1_g4->SetBranchAddress("z",&pz_g4);
   t1_g4->SetBranchAddress("E",&E_g4);

   TH1F *hx   = new TH1F("hx","Ex distribution",25,0,25);
   TH1F *hy   = new TH1F("hy","Ey distribution",25,0,25);
   TH1F *hz   = new TH1F("hz","Ez distribution",25,0,25);
   TH1F *hx_g4   = new TH1F("hx_g4","Ex distribution",25,0,25);
   TH1F *hy_g4   = new TH1F("hy_g4","Ey distribution",25,0,25);
   TH1F *hz_g4   = new TH1F("hz_g4","Ez distribution",25,0,25);
   TH1F *hx_half   = new TH1F("hx_half","Ex distribution",25,0,25);
   TH1F *hy_half   = new TH1F("hy_half","Ey distribution",25,0,25);
   TH1F *hz_half   = new TH1F("hz_half","Ez distribution",25,0,25);
   TH1F *hx_half_g4   = new TH1F("hx_half_g4","Ex distribution",25,0,25);
   TH1F *hy_half_g4   = new TH1F("hy_half_g4","Ey distribution",25,0,25);
   TH1F *hz_half_g4   = new TH1F("hz_half_g4","Ez distribution",25,0,25);
   TH3D *gan = new TH3D("gan","E distribution", 25,0,25,25,0,25,25,0,25);
   TH3D *g4 = new TH3D("g4","E distribution", 25,0,25,25,0,25,25,0,25);
   hx->Sumw2();
   hy->Sumw2();
   hz->Sumw2();
   hx_g4->Sumw2();
   hy_g4->Sumw2();
   hz_g4->Sumw2();
   hx_half->Sumw2();
   hy_half->Sumw2();
   hz_half->Sumw2();
   hx_half_g4->Sumw2();
   hy_half_g4->Sumw2();
   hz_half_g4->Sumw2();


   int SLICE = 9;
   TH1F* hx_short[SLICE]; 
   TH1F* hy_short[SLICE]; 
   TH1F* hx_g4_short[SLICE]; 
   TH1F* hy_g4_short[SLICE]; 

   for (int i=0;i<SLICE;i++) {
      int nbins = 25 - 2*i;
      int low_bin = i;
      int high_bin = 25 -i;
      char hx_name[10];
      char hy_name[10];
      char hx_g4_name[20];
      char hy_g4_name[20];
      sprintf(hx_name,"%s%d","hx_short",i);
      sprintf(hy_name,"%s%d","hy_short",i);
      sprintf(hx_g4_name,"%s%d","hx_g4_short",i);
      sprintf(hy_g4_name,"%s%d","hy_g4_short",i);
      hx_short[i] = new TH1F(hx_name,"",nbins,low_bin,high_bin);
      hy_short[i] = new TH1F(hy_name,"",nbins,low_bin,high_bin);
      hx_g4_short[i] = new TH1F(hx_g4_name,"",nbins,low_bin,high_bin);
      hy_g4_short[i] = new TH1F(hy_g4_name,"",nbins,low_bin,high_bin);
      hx_short[i]->Sumw2();
      hy_short[i]->Sumw2();
      hx_g4_short[i]->Sumw2();
      hy_g4_short[i]->Sumw2();
   }

   int FLOOR = 9;
   int CELLX = 5;
   int CELLY = 5;
   int CELLZ = 5;
   TH1F* hE_gan[CELLX][CELLY][CELLZ]; 
   TH1F* hE_g4[CELLX][CELLX][CELLZ];
   int nEbins = 60;
   int low_Ebin = 0;
   int high_Ebin = 180;
 
   for (int i=0;i<CELLX;i++) {
      for (int j=0;j<CELLY;j++) {
         for (int k=0;k<CELLZ;k++) {
              char hE_name[12];
              char hE_g4_name[11];
              //sprintf(hE_name,"%s%d%s%d%s%d","hE_gan_",i,"_",j,"_",k);
              //sprintf(hE_g4_name,"%s%d%s%d%s%d","hE_g4_",i,"_",j,"_",k);
              sprintf(hE_name,"%s%d%d%d","hE_gan_",i,j,k);
              sprintf(hE_g4_name,"%s%d%d%d","hE_g4_",i,j,k);
              //cout<<hE_name<<endl;
              //cout<<hE_g4_name<<endl;
              hE_gan[i][j][k] = new TH1F(hE_name,"",nEbins,low_Ebin,high_Ebin);
              hE_g4[i][j][k] = new TH1F(hE_g4_name,"",nEbins,low_Ebin,high_Ebin);
              hE_gan[i][j][k]->Sumw2();
              hE_g4[i][j][k]->Sumw2();
          }
      }
   }

   Long64_t nentries = t1->GetEntries();
   cout<<"GAN events : "<<nentries<<endl;
   for (Long64_t i=0;i<nentries;i++) {
      t1->GetEntry(i);
      cout<<"entries OK"<<std::endl;
      cout<<"vec sizes "<<E->size()<<" "<<px->size()<<" "<<py->size()<<" "<<pz->size()<<std::endl;
      //int NTOT=nentries*25*25*25;
      for (int ie = 0;ie<E->size();ie++) {
         
        if (E->at(ie)>0.001) {
            hx->Fill(px->at(ie),E->at(ie));
            hy->Fill(py->at(ie),E->at(ie));
            hz->Fill(pz->at(ie),E->at(ie));
            for (int is=0;is<SLICE;is++) {
                hx_short[is]->Fill(px->at(ie),E->at(ie));
                hy_short[is]->Fill(py->at(ie),E->at(ie));
            }
         }
         if (pz->at(ie)==12 &&E->at(ie)>0.001) {
            hx_half->Fill(px->at(ie),E->at(ie));
            hy_half->Fill(py->at(ie),E->at(ie));
            hz_half->Fill(pz->at(ie),E->at(ie));
         }
         if (pz->at(ie)>FLOOR && pz->at(ie)<(FLOOR+CELLZ+1) && px->at(ie)>FLOOR && px->at(ie)<(FLOOR+CELLX+1) &&py->at(ie)>FLOOR && py->at(ie)<(FLOOR+CELLY+1) &&E->at(ie)>0.001) {
             int idx = px->at(ie) -FLOOR - 1;
             int idy = py->at(ie) -FLOOR - 1;
             int idz = pz->at(ie) -FLOOR - 1;
             hE_gan[idx][idy][idz]->Fill(E->at(ie));
         }
      }
  
   }

   Long64_t nentries_g4 = t1_g4->GetEntries();
   cout<<"G4 entries: "<<nentries_g4<<" GAN nentries "<<nentries<< endl;
   for (Long64_t j=0;j<nentries_g4;j++) {
      t1_g4->GetEntry(j);
      for (int je = 0;je<E_g4->size();je++) {
         if (E_g4->at(je)>0) {
            hx_g4->Fill(px_g4->at(je),E_g4->at(je));
            hy_g4->Fill(py_g4->at(je),E_g4->at(je));
            hz_g4->Fill(pz_g4->at(je),E_g4->at(je));
            for (int i=0;i<SLICE;i++) {
               hx_g4_short[i]->Fill(px_g4->at(je),E_g4->at(je));
               hy_g4_short[i]->Fill(py_g4->at(je),E_g4->at(je));
            }
         }
         if (pz_g4->at(je)==12 && E_g4->at(je)>0) {
            hx_half_g4->Fill(px_g4->at(je),E_g4->at(je));
            hy_half_g4->Fill(py_g4->at(je),E_g4->at(je));
            hz_half_g4->Fill(pz_g4->at(je),E_g4->at(je));
         }
         if (pz_g4->at(je)>FLOOR && pz_g4->at(je)<(FLOOR+CELLZ+1) && px_g4->at(je)>FLOOR && px_g4->at(je)<(FLOOR+CELLX+1) &&py_g4->at(je)>FLOOR && py_g4->at(je)<(FLOOR+CELLY+1) &&E_g4->at(je)>0.001) { 
            int idx = px_g4->at(je) -FLOOR - 1;
            int idy = py_g4->at(je) -FLOOR - 1;
            int idz = pz_g4->at(je) -FLOOR - 1;
            hE_g4[idx][idy][idz]->Fill(E_g4->at(je));
         }
      }
   }
  //hx->Scale(hx_g4->Integral(0,24)/hx->Integral(0,24));
  //hy->Scale(hy_g4->Integral(0,24)/hy->Integral(0,24));
 // hz->Scale(hz_g4->Integral(0,24)/hz->Integral(0,24));


  hx->SetLineColor(2);
  hy->SetLineColor(2);
  hz->SetLineColor(2);
  TCanvas *c1=new TCanvas("c1","",200,10,700,500);
  c1->cd();
  hx->SetStats(0);
  hx->Draw();
  hx_g4->Draw("SAME");
/*
  double probx = hx->Chi2Test(hx_g4,"WWPUF");
  double kolx = hx->KolmogorovTest(hx_g4,"D");
  c1->SaveAs("hx_lin_50epochs.C");
  c1->SaveAs("hx_lin_50epochs.pdf");
*/
  TCanvas *c2=new TCanvas("c2","",200,10,700,500);
  c2->cd();
  hy->SetStats(0);
  hy->Draw();
  hy_g4->Draw("SAME");
/*
  double proby = hy->Chi2Test(hy_g4,"WWPUF");
  double koly = hy->KolmogorovTest(hy_g4,"D");
  c2->SaveAs("hy_lin_50epochs.C");
*/
  TCanvas *c3=new TCanvas("c3","",200,10,700,500);
  c3->cd();
  hz->Draw();
  hz_g4->Draw("SAMES");
/*
  double probz = hz->Chi2Test(hz_g4,"WWPUF");
  double kolz = hz->KolmogorovTest(hz_g4,"D");
  c3->SaveAs("hz_ele.C");
  c3->SaveAs("hz_ele.pdf");
*/
  std::cout<<""<<std::endl;
  std::cout<<""<<std::endl;
  std::cout<<"=================================================="<<std::endl;
  std::cout<<""<<std::endl;
  std::cout<<""<<std::endl;

  for (int i=0;i<SLICE;i++) {
     double ppx = hx_short[i]->Chi2Test(hx_g4_short[i],"WWp");
     double ppy = hy_short[i]->Chi2Test(hy_g4_short[i],"WWp");
  }

  TCanvas *c4=new TCanvas("c4","",200,10,700,500);
  c4->cd();
  double probE = hE_gan[2][2][2]->Chi2Test(hE_g4[2][2][2],"WWPUF");
  double kolE = hE_gan[2][2][2]->KolmogorovTest(hE_g4[2][2][2],"D");
  hE_gan[2][2][2]->Draw();
  hE_g4[2][2][2]->Draw("SAMES");

  TFile *fE =new TFile("Ehistos_gendiscr000_E.root","RECREATE");
  
  TGraphAsymmErrors* Mgr = new TGraphAsymmErrors();
  TGraphAsymmErrors* Sgr = new TGraphAsymmErrors();
  int npoints = 0;
  for (int i=0;i<CELLX;i++) 
     for (int j=0;j<CELLY;j++) 
        for (int k=0;k<CELLZ;k++) {
           hE_gan[i][j][k]->SetLineColor(2);
           float Mgan = hE_gan[i][j][k]->GetMean();
           float Sgan = hE_gan[i][j][k]->GetStdDev();
           float Mgan_err = hE_gan[i][j][k]->GetMeanError();
           float Sgan_err = hE_gan[i][j][k]->GetStdDevError();
           float Mg4 = hE_g4[i][j][k]->GetMean();
           float Sg4 = hE_g4[i][j][k]->GetStdDev();
           float Mg4_err = hE_g4[i][j][k]->GetMeanError();
           float Sg4_err = hE_g4[i][j][k]->GetStdDevError();
           float Mratio = 0;
           float Mratio_err = 0;
           if (Mg4 !=0 && Mgan !=0) {
                 Mratio = Mgan/Mg4;
                 Mratio_err = Mratio*sqrt((Mgan_err*Mgan_err/(Mgan*Mgan)) + (Mg4_err*Mg4_err/(Mg4*Mg4)));
           }  
           float Sratio = 0;
           float Sratio_err = 0;
           if (Sg4 !=0 && Sgan !=0) {
                 Sratio = 1.2*Sgan/Sg4;
                 Sratio_err = Sratio*sqrt((Sgan_err*Sgan_err/(Sgan*Sgan)) + (Sg4_err*Sg4_err/(Sg4*Sg4)));
           }
           Mgr->SetPoint(npoints, npoints, Mratio);
           Mgr->SetPointEYhigh(npoints,Mratio_err); 
           Mgr->SetPointEYlow(npoints,Mratio_err); 
           Sgr->SetPoint(npoints, npoints,Sratio);
           Sgr->SetPointEYhigh(npoints,Sratio_err); 
           Sgr->SetPointEYlow(npoints,Sratio_err); 
           npoints++;

           hE_gan[i][j][k]->Write();
           hE_g4[i][j][k]->Write();
    
        }
        TCanvas *c5=new TCanvas("c5","",200,10,700,500);
        c5->cd();
         Mgr->SetMarkerColor(2);
         Mgr->SetMarkerStyle(21);
         Sgr->SetMarkerColor(3);
         Sgr->SetMarkerStyle(21);
        Mgr->Draw("ALP");
        Mgr->SetName("Mgr");
        c5->SaveAs("Mgr.C");
       TCanvas *c6=new TCanvas("c6","",200,10,700,500);
       c6->cd();
        Sgr->Draw("ALP");
        Sgr->SetName("Sgr");
        c6->SaveAs("Sgr.C");
        Mgr->Write();
        Sgr->Write();
   hx->Write();
   hy->Write();
   hz->Write();
}
