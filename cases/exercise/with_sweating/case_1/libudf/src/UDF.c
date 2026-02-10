#include "udf.h"
#define dblood Get_Input_Parameter("d_blood")
#define perfhead 0.0083333
#define perforgan 0.001266
#define cblood Get_Input_Parameter("c_blood")
#define metabhead Get_Input_Parameter("metab_head")
#define metaborgan Get_Input_Parameter("metab_organ")
#define qext 0.0
#define vblood Get_Input_Parameter("v_blood")
#define heattransfercoefficient Get_Input_Parameter("h")
#define evaporativehtc 102.79

#define ambienttemperature Get_Input_Parameter("T_amb") - 273.15
// #define HR Get_Input_Parameter("exp1")


real core_tempa;
real perfmuscle=0.0005;
real metabmuscle= 553.5;
real sweatingcal;
real convection = 0.;


DEFINE_SOURCE(head_source,c,t,dS,eqn) 
{   
  real tblood_p1,ttissue,source;
  ttissue = C_T(c,t) - 273.15;
  tblood_p1 = C_UDMI(c,t,0) - 273.15;
  /*Net Source Term, includes metabolic rate and perfusion rate */
  source = metabhead+(dblood*cblood*perfhead*(tblood_p1-ttissue));
  /*Term below is just to aid convergence, does not affect heat source*/
  dS[eqn] = -dblood*cblood*perfhead;
  return source;
}


DEFINE_SOURCE(organ_source,c,t,dS,eqn) 
{   
  real tblood_p1,ttissue,source;
  ttissue = C_T(c,t) - 273.15;
  tblood_p1 = C_UDMI(c,t,0) - 273.15;
  /*Net Source Term, includes metabolic rate and perfusion rate */
  source = metaborgan+(dblood*cblood*perforgan*(tblood_p1-ttissue));
  /*Term below is just to aid convergence, does not affect heat source*/
  dS[eqn] = -dblood*cblood*perforgan;
  return source;
}

DEFINE_SOURCE(muscle_source,c,t,dS,eqn) 
{   
  real tblood_p1,ttissue,source;
  ttissue = C_T(c,t) - 273.15;
  tblood_p1 = C_UDMI(c,t,0) - 273.15;
  /*Net Source Term, includes metabolic rate and perfusion rate */
  source = metabmuscle+(dblood*cblood*perfmuscle*(tblood_p1-ttissue));
  /*Term below is just to aid convergence, does not affect heat source*/
  dS[eqn] = -dblood*cblood*perfmuscle;
  return source;
}




DEFINE_EXECUTE_AT_END(blood_temperature)
{
 Domain *d;
 Thread *t;
 real sum_tisstemp=0;
 real metab_ramp_time=300;
 real sum_tisstemph=0.;
 real sum_tisstempo=0.;
 real sum_tisstempm=0.;
 real vbody=0;
 real vbody_head=0.;
 real core_temp=0.;
 real vbody_organ=0.;
 real vbody_muscle=0.;
 real avg_tisstemp,wavg_tisstemp, tblood_p, tblood_p1;
 real wavg_perf;
 cell_t c;
 real tstep,time;
 real term1, term2, term3;
 int count=0;
 int ZONE_ID_head=10;
 Thread *t_head;
 int ZONE_ID_organ=8;
 Thread *t_organ;
 int ZONE_ID_muscle=9;
 Thread *t_muscle;
 tstep = CURRENT_TIMESTEP;
 time=CURRENT_TIME;
 
 
 d=Get_Domain(1);

 t_head=Lookup_Thread(d,ZONE_ID_head);
 t_organ=Lookup_Thread(d,ZONE_ID_organ);
 t_muscle=Lookup_Thread(d,ZONE_ID_muscle);

Message("T_amb: %g\n", ambienttemperature);

/*calculation of average temperature*/

 thread_loop_c(t,d)
 {
   begin_c_loop(c,t)
   {
     sum_tisstemp += C_T(c,t)*C_VOLUME(c,t);
     vbody += C_VOLUME (c,t);
   }
   end_c_loop(c,t)
 }

   begin_c_loop(c,t_head)
   {
     sum_tisstemph += C_T(c,t_head)*C_VOLUME(c,t_head)*perfhead;
     vbody_head += C_VOLUME (c,t_head);
   }
   end_c_loop(c,t_head)
   begin_c_loop(c,t_organ)
   {
     sum_tisstempo += C_T(c,t_organ)*C_VOLUME(c,t_organ)*perforgan;
     vbody_organ += C_VOLUME (c,t_organ);
   }
   end_c_loop(c,t_organ)
   begin_c_loop(c,t_muscle)
   {
     sum_tisstempm += C_T(c,t_muscle)*C_VOLUME(c,t_muscle)*perfmuscle;
     vbody_muscle += C_VOLUME (c,t_muscle);
   }
   end_c_loop(c,t_muscle)

sum_tisstemp = PRF_GRSUM1(sum_tisstemp);
sum_tisstemph = PRF_GRSUM1(sum_tisstemph);
sum_tisstempo = PRF_GRSUM1(sum_tisstempo);
sum_tisstempm = PRF_GRSUM1(sum_tisstempm);
vbody_organ = PRF_GRSUM1(vbody_organ);
vbody_muscle = PRF_GRSUM1(vbody_muscle);
vbody_head = PRF_GRSUM1(vbody_head);
vbody= PRF_GRSUM1(vbody);

wavg_perf=(perfhead*vbody_head+perforgan*vbody_organ+perfmuscle*vbody_muscle)/vbody;
wavg_tisstemp=((sum_tisstemph+sum_tisstempo+sum_tisstempm)/(wavg_perf*vbody)) -273.15;
core_temp=(sum_tisstempo/(vbody_organ*perforgan)) -273.15;
core_tempa=core_temp;
avg_tisstemp=(sum_tisstemp/vbody) - 273.15;

Message("wavg_perf: %g\n",wavg_perf);
Message("avg_tisstemp: %g, core_temp: %g, wavg_tisstemp: %g\n",avg_tisstemp,core_temp,wavg_tisstemp);
Message("Time: %g, Iteration: %d\n", time, N_ITER);


if(metab_ramp_time>time)
{
perfmuscle=(time*0.000010408)+0.0005;
}
else
{
perfmuscle= 0.0036224;
}

if(metab_ramp_time>time)
{
metabmuscle=(time*11.5219)+Get_Input_Parameter("metab_muscle");
}
else
{
metabmuscle= 4010.07;
}


   Message("Metabolic rate: %g\n",metabmuscle);
   Message("Perfusion: %g\n",perfmuscle);
   Message("Internal Core: %g\n",core_tempa);
   Message("Time: %g\n",time);

/*Extaraction of old blood temperature, saving of wtd avg tissue temp*/

thread_loop_c(t,d)
{
  begin_c_loop(c,t)
  {
    C_UDMI(c,t,0) = 310.15;
    tblood_p=C_UDMI(c,t,0) -273.15;
    C_UDMI(c,t,1) = wavg_tisstemp + 273.15;
    C_UDMI(c,t,2) = avg_tisstemp + 273.15;
    C_UDMI(c,t,3) = core_temp + 273.15;
    C_UDMI(c,t,4) = sweatingcal;
  }
  end_c_loop(c,t)
}
  //  Message("wavg_perf: %g\n",wavg_perf);
  //  Message("avg_tisstemp: %g, core_temp: %g, wavg_tisstemp: %g\n",avg_tisstemp,core_temp,wavg_tisstemp);


/*Calculation of new blood temperature*/

if (time > 0.0)
{
   term1=(qext*tstep)/(dblood*cblood*vblood);
   term2=wavg_tisstemp*((dblood*cblood*wavg_perf*vbody*tstep)/(dblood*cblood*vblood));
   term3=-tblood_p*((dblood*cblood*wavg_perf*vbody*tstep)/(dblood*cblood*vblood));
   tblood_p1 = (term1+tblood_p+term2 + term3);
}
else
{
   
   tblood_p1 = tblood_p; // steady
}


/*printf(" term1,term2,term3: %g,%g,%g\n", term1,term2,term3);*/
 thread_loop_c(t,d)
 {
   begin_c_loop(c,t)
   {
      C_UDMI(c,t,0) = tblood_p1 + 273.15;
   }
   end_c_loop(c,t)
 } 
// Message(" New Blood T deg C at time: %g,%g\n", tblood_p1,time);
/*fflush(stdout);*/
   Message("Sweating_cal: %g\n",sweatingcal);
   Message("Convection: %g\n",convection);
}


DEFINE_PROFILE(convection_heat_transfer,t,i)

{

real tskin=0.;

real sweating=0.;

face_t f;

real timeh=300;
real time;
real percentage;
time=CURRENT_TIME;
begin_f_loop(f,t)

{

tskin = F_T(f,t) - 273.15;
  //  Message("F_t: %g\n",F_T(f,t));
  //  Message("t_skin: %g\n",tskin);

convection = heattransfercoefficient*(ambienttemperature-tskin);


if(tskin<ambienttemperature)
{
sweating=0;
}
else if(core_tempa + 273.15>=310.38 && core_tempa + 273.15 <310.58)
{
percentage=(5*core_tempa)-186.15;
   Message("core_tempa: %g\n", core_tempa );
   Message("percentage: %g\n", percentage );

sweating=-percentage*0.1333*evaporativehtc*(((1.92*tskin)-25.3)-(23.8*0.50));
sweatingcal=sweating;
}
else if(core_tempa + 273.15>=310.58)
{
percentage=1;
   Message("core_tempa: %g\n", core_tempa );
   Message("percentage: %g\n", percentage );

sweating=-percentage*0.1333*evaporativehtc*(((1.92*tskin)-25.3)-(23.8*0.50));
sweatingcal=sweating;
}
else 
{
sweating=0;
}



F_PROFILE(f,t,i) = convection+sweating;
  //  Message("conv+sweat: %g\n", F_PROFILE(f,t,i) );

}

end_f_loop(f,t)

}

/* Initialize blood temperature to 310.15 K in all cells */
DEFINE_INIT(initialize_blood_temp, d)
{
    Thread *t;
    cell_t c;
    
    /* Loop over all cell threads in the domain */
    thread_loop_c(t,d)
    {
        /* Loop over all cells in the thread */
        begin_c_loop(c,t)
        {
            C_UDMI(c,t,0) = 310.15;  /* Set initial blood temperature to 310.15 K */
        }
        end_c_loop(c,t)
    }
}
