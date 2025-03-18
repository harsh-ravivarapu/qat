function [beta_vec, EI, CTX_workspace] = rl_BGM_step_SMC_pulse_python_step(tmax,IT,pd,corstim,freq,pprofile,len,dt,stride,windowlen)
%tmax - time the simulation will run (ms)
%IT - iteration number (trial no)
%pd - 0(normal/healthy condition), 1(Parkinson's disease(PD) condition)
%corstim (cortical stimulation) - 0(off), 1(on)
%freq - frequency used for action/IDBS stimulation (Hz)
%pprofile - pulse profile/action vector: [] if random
%len - amount of time the outputs should represent (ms)
%dt - amount of time between last step of environment and the following
%step (ms) - default 0.01ms
%stride - defines stride for calculating beta from generated data (ms)
%windowlen - amount of data used to calculate beta (ms) - should be larger
%than 200ms
% ADDED STRUCT FUNCTIONALITY FOR PARALLEL
% CAN BE USED FOR DIFFERENT SIZE WINDOW
% UPDATED SO TIME STEP DT CAN BE CHANGED
% stride defines how far the window slides over (in ms) Default is 1
% removed error index
% Stride and Window edited
% UPDATED TO HAVE A ANY SIZE SECOND WINDOW
% UPDATED TO HANDLE CHANGING TIME STEP
% EI removed
% Stride and Window edited
% updated creatdbs to creatdbs2

% Random
rng shuffle;

n = 10;             % number of neurons in each nucleus (orig 10)
%dt = 0.01;          % ms
t=0:dt:tmax;        % time vector used for environment


% DBS Parameters

PW = 0.3;           % ms [DBS pulse width]
amplitude = 300;    % nA/cm2 [DBS current amplitude], originally 300
% freqs=0:5:200;      % DBS frequency in Hz

if (class(freq)=='string')
    pattern = randi([0 200],1,1);
else
    pattern = freq;
end


% Create DBS Current   (currently it is on STN (check Idbs))

if pattern==0 
  % no stimulation  
  Idbs_step=zeros(1,length(t)); 
else
  % IDBS (action) is generated using pprofile
  [Idbs_step, ~]=creatdbs2(pattern,tmax,dt,PW,amplitude,pprofile); 
end


% Create Cortical Stimulus Pulse

if corstim==1
  % Create Cortical Stimulus Pulse
  Iappco_step=zeros(1,length(t));
  Iappco_step((1000/dt):((1000+0.3)/dt))=350;
  
else
  % No Cortical Stimulus Pulse
  Iappco_step=zeros(1,length(t));
  
end


% Run CTX-BG-TH Network Model
[vgpi,vTH,SMC_pulse,CTX_workspace] = CTX_BG_TH_network(pd,corstim,tmax,dt,n,Idbs_step,Iappco_step,len,windowlen); %(pd,corstim,tmax,dt,n,Idbs_step,Iappco_step,len,CTX_workspace,IT)% [vgpi,vTH,SMC] = CTX_BG_TH_network(pd,corstim,tmax,dt,n,Idbs,Iappco,SMC_pulse);


% Calculate GPi pathological low-frequency oscillatory power
dt1=dt*10^-3;
params.Fs = 1/dt1; %Hz
params.fpass = [1 100];
params.tapers = [3 5];
params.trialave = 1;


% calculate beta and EI
[beta_vec, EI] = sliding_window_2(vgpi,vTH,SMC_pulse,params,n,tmax,CTX_workspace.timespike,dt,stride,windowlen);



 % save results 
 if pd==0
    name = [num2str(IT) 'con' num2str(pattern) 'rs.mat'];
 else
    name = [num2str(IT) 'pd' num2str(pattern) 'rs.mat'];
 end
 save(name,'beta_vec', 'Idbs_step', 'EI')

% quit

end


function [Idbs_step, intermediate] = creatdbs2(pattern,tmax,dt,PW,amplitude,pprofile)
% second iteration of the conversion of pprofile (action) to IDBS vector
% used directly in environment

t=0:dt:tmax; % Length of array = tmax/dt
pulse_w = PW/dt; % Pulse width in index

% Create initial DBS array and pulse array (iD is the amplitude)
Idbs_step=zeros(1,length(t)); 
iD=amplitude;            % nA/cm2 [DBS current amplitude]
pulse=iD*ones(1,pulse_w);

if isempty(pprofile)
    impulse = zeros(1,pattern); % list of impulse locations.
    inst = pattern;
    isi = 1000/inst;
    ipi=round(isi*1/dt);
    i=1;
    n_im = 1;
    while i<length(t)
        j=randi([i i+ipi],1,1);
        if j > length(t)
            break
        end
        Idbs_step(j:j+pulse_w-1)=pulse;
        i=i+round(isi*1/dt);
        impulse(n_im)=j;
        n_im = n_im+1;
    end
else
    % Find how many zeros go between each index in pprofile
    amt_zeros = round((length(Idbs_step))/length(pprofile));

    % set variables to 1
    i = 1;
    j = 1;
    
    % loop through pprofile to create the IDBS
    while j <= length(pprofile)
        % effectively adds whatever was in pprofile into IDBS
        % and fills space between with zeros 
        % done to use find() later on in the function
        Idbs_step(i) = pprofile(j);
%        Idbs_step(i:i+amt_zeros) = pprofile(j);
        % move forward in the pprofile and IDB
        i = i+amt_zeros; 
        j = j+1;
    end
    % return this intermediate to verify the system works
    intermediate = Idbs_step;

    % convert the intermediate to something that can be used by the
    % environment
    impulse = find(Idbs_step);
    if impulse(end) > length(Idbs_step)-pulse_w
        impulse(end) = length(Idbs_step)-pulse_w;
    end
    for i = 1:length(impulse)
        Idbs_step(impulse(i):impulse(i)+pulse_w-1)=pulse;
    end
end
end


function [Idbs_step, impulse]=creatdbs(pattern,tmax,dt,PW,amplitude,pprofile)
%converts pprofile to the IDBS vector used directly in the environment

% Define time vector
t=0:dt:tmax; % Length of array = tmax/dt
pulse_w = PW/dt; % Pulse width in index

% Create initial DBS array and pulse array (iD is the amplitude)
Idbs_step=zeros(1,length(t)); 
iD=amplitude;            % nA/cm2 [DBS current amplitude]
pulse=iD*ones(1,pulse_w);

% Loop through Idbs array and embed pulse array at random intervals while
% keeping the one second average frequence equal to the defined parameter
% above.

if isempty(pprofile)
    impulse = zeros(1,pattern); % list of impulse locations.
    inst = pattern;
    isi = 1000/inst;
    ipi=round(isi*1/dt);
    i=1;
    n_im = 1;
    while i<length(t)
        j=randi([i i+ipi],1,1);
        if j > length(t)
            break
        end
        Idbs_step(j:j+pulse_w-1)=pulse;
        i=i+round(isi*1/dt);
        impulse(n_im)=j;
        n_im = n_im+1;
    end

else
    impulse = find(pprofile);
    if length(pprofile) ~= length(Idbs_step)
        impulse = impulse.*round(length(Idbs_step)/length(pprofile));
    end

    if impulse(end) > length(Idbs_step)-pulse_w
        impulse(end) = length(Idbs_step)-pulse_w;
    end

    for i = 1:length(impulse)
        Idbs_step(impulse(i):impulse(i)+pulse_w-1)=pulse;
    end
end
end

function [vgpi,vThal,SMC_pulse,CTX_workspace] = CTX_BG_TH_network(new_pd,new_corstim,new_tmax,new_dt,new_n,new_Idbs_step,new_Iappco_step,new_len,windowlen) %(pd,corstim,tmax,dt,n,Idbs_step,Iappco_step,len,CTX_workspace,IT)%[vgpi,vTH,SMC] = CTX_BG_TH_network(pd,corstim,tmax,dt,n,Idbs,Iappco,SMC_pulse);
%Environment
    load("CTX_workspace");
    
    % Load Struct
	a2 = CTX_workspace.a2;
    A2 = CTX_workspace.A2;
    a3 = CTX_workspace.a3;
    a4 = CTX_workspace.a4;
    ace = CTX_workspace.ace;
    aci = CTX_workspace.aci;
    ae = CTX_workspace.ae;
    ai = CTX_workspace.ai;
    all = CTX_workspace.all;
    alp = CTX_workspace.alp;
    b2 = CTX_workspace.b2;
    B2 = CTX_workspace.B2;
    be = CTX_workspace.be;
    bi = CTX_workspace.bi;
    bll = CTX_workspace.bll;
    c2 = CTX_workspace.c2;
    C2 = CTX_workspace.C2;
    CA2 = CTX_workspace.CA2;
    CA3 = CTX_workspace.CA3;
    CA4 = CTX_workspace.CA4;
    Cao = CTX_workspace.Cao;
    CAsn2 = CTX_workspace.CAsn2;
    ce = CTX_workspace.ce;
    ci = CTX_workspace.ci;
    cll = CTX_workspace.cll;
    Cm = CTX_workspace.Cm;
    con = CTX_workspace.con;
    const = CTX_workspace.const;
    const1 = CTX_workspace.const1;
    const2 = CTX_workspace.const2;
    d1 = CTX_workspace.d1;
    D1 = CTX_workspace.D1;
    d2 = CTX_workspace.d2;
    D2 = CTX_workspace.D2;
    de = CTX_workspace.de;
    di = CTX_workspace.di;
    dll = CTX_workspace.dll;
    %dt = CTX_workspace.dt;
    Eca = CTX_workspace.Eca;
    Ecasn = CTX_workspace.Ecasn;
    Ek = CTX_workspace.Ek;
    El = CTX_workspace.El;
    ell = CTX_workspace.ell;
    Em = CTX_workspace.Em;
    Ena = CTX_workspace.Ena;
    Esyn = CTX_workspace.Esyn;
    Et = CTX_workspace.Et;
    F = CTX_workspace.F;
    fa = CTX_workspace.fa;
    fg = CTX_workspace.fg;
    fll = CTX_workspace.fll;
    fn = CTX_workspace.fn;
    fstngpea = CTX_workspace.fstngpea;
    fstngpen = CTX_workspace.fstngpen;
    ga1 = CTX_workspace.ga1;
    gahp = CTX_workspace.gahp;
    gca = CTX_workspace.gca;
    gcak = CTX_workspace.gcak;
    gcordrstr = CTX_workspace.gcordrstr;
    gcorindrstr = CTX_workspace.gcorindrstr;
    gcorsna = CTX_workspace.gcorsna;
    gcorsnn = CTX_workspace.gcorsnn;
    gei = CTX_workspace.gei;
    ggaba = CTX_workspace.ggaba;
    ggege = CTX_workspace.ggege;
    ggesn = CTX_workspace.ggesn;
    ggigi = CTX_workspace.ggigi;
    ggith = CTX_workspace.ggith;
    gie = CTX_workspace.gie;
    gk = CTX_workspace.gk;
    gl = CTX_workspace.gl;
    gL = CTX_workspace.gL;
    gll = CTX_workspace.gll;
    gm = CTX_workspace.gm;
    gna = CTX_workspace.gna;
    gpeak = CTX_workspace.gpeak;
    gpeak1 = CTX_workspace.gpeak1;
    gsngea = CTX_workspace.gsngea;
    gsngen = CTX_workspace.gsngen;
    gsngi = CTX_workspace.gsngi;
    gstrgpe = CTX_workspace.gstrgpe;
    gstrgpi = CTX_workspace.gstrgpi;
    gt = CTX_workspace.gt;
    gthcor = CTX_workspace.gthcor;
    h1 = CTX_workspace.h1;
    H1 = CTX_workspace.H1;
    h2 = CTX_workspace.h2;
    H2 = CTX_workspace.H2;
    h3 = CTX_workspace.h3;
    H3 = CTX_workspace.H3;
    h4 = CTX_workspace.h4;
    H4 = CTX_workspace.h4;
    h5 = CTX_workspace.h5;
    h6 = CTX_workspace.h6;
    hll = CTX_workspace.hll;
    i = CTX_workspace.i;
    Ia2 = CTX_workspace.Ia2;
    Iahp3 = CTX_workspace.Iahp3;
    Iahp4 = CTX_workspace.Iahp4;
    Iappco = CTX_workspace.Iappco;
    Iappgpe = CTX_workspace.Iappgpe;
    Iappgpi = CTX_workspace.Iappgpi;
    Iappth = CTX_workspace.Iappth;
    Ica3 = CTX_workspace.Ica3;
    Ica4 = CTX_workspace.Ica4;
    Icak2 = CTX_workspace.Icak2;
    Icorsnampa = CTX_workspace.Icorsnampa;
    Icorsnnmda = CTX_workspace.Icorsnnmda;
    Icorstr5 = CTX_workspace.Icorstr5;
    Icorstr6 = CTX_workspace.Icorstr6;
    Idbs = CTX_workspace.Idbs;
    Iei = CTX_workspace.Iei;
    Igaba5 = CTX_workspace.Igaba5;
    Igaba6 = CTX_workspace.Igaba6;
    Igege = CTX_workspace.Igege;
    Igesn = CTX_workspace.Igesn;
    Igigi = CTX_workspace.Igigi;
    Igith = CTX_workspace.Igith;
    Iie = CTX_workspace.Iie;
    Ik1 = CTX_workspace.Ik1;
    Ik2 = CTX_workspace.Ik2;
    Ik3 = CTX_workspace.Ik3;
    Ik4 = CTX_workspace.Ik4;
    Ik5 = CTX_workspace.Ik5;
    Ik6 = CTX_workspace.Ik6;
    Il1 = CTX_workspace.Il1;
    Il2 = CTX_workspace.Il2;
    IL2 = CTX_workspace.IL2;
	Il3 = CTX_workspace.Il3;
	Il4 = CTX_workspace.Il4;
	Il5 = CTX_workspace.Il5;
	Il6 = CTX_workspace.Il6;
	ill = CTX_workspace.ill;
	Im5 = CTX_workspace.Im5;
	Im6 = CTX_workspace.Im6;
	Ina1 = CTX_workspace.Ina1;
	Ina2 = CTX_workspace.Ina2;
	Ina3 = CTX_workspace.Ina3;
	Ina4 = CTX_workspace.Ina4;
	Ina5 = CTX_workspace.Ina5;
	Ina6 = CTX_workspace.Ina6;
	Isngeampa = CTX_workspace.Isngeampa;
	Isngenmda = CTX_workspace.Isngenmda;
	Isngi = CTX_workspace.Isngi;
	Istrgpe = CTX_workspace.Istrgpe;
	Istrgpi = CTX_workspace.Istrgpi;
	It1 = CTX_workspace.It1;
	It2 = CTX_workspace.It2;
	It3 = CTX_workspace.It3;
	It4 = CTX_workspace.It4;
	Ithcor = CTX_workspace.Ithcor;
	j = CTX_workspace.j;
	jll = CTX_workspace.jll;
	k1 = CTX_workspace.k1;
	kca = CTX_workspace.kca;
	Kca = CTX_workspace.Kca;
	kll = CTX_workspace.kll;
	lll = CTX_workspace.lll;
	m1 = CTX_workspace.m1;
	m2 = CTX_workspace.m2;
	M2 = CTX_workspace.M2;
	m3 = CTX_workspace.m3;
	m4 = CTX_workspace.m4;
	m5 = CTX_workspace.m5;
	m6 = CTX_workspace.m6;
	mll = CTX_workspace.mll;
	n2 = CTX_workspace.n2;
	N2 = CTX_workspace.N2;
	n3 = CTX_workspace.n3;
	N3 = CTX_workspace.N3;
	n4 = CTX_workspace.n4;
	N4 = CTX_workspace.N4;
	n5 = CTX_workspace.n5;
	n6 = CTX_workspace.n6;
	nll = CTX_workspace.nll;
	oll = CTX_workspace.oll;
	p1 = CTX_workspace.p1;
	p2 = CTX_workspace.p2;
	P2 = CTX_workspace.P2;
	p5 = CTX_workspace.p5;
	p6 = CTX_workspace.p6;
	q2 = CTX_workspace.q2;
	Q2 = CTX_workspace.Q2;
	R = CTX_workspace.R;
	r1 = CTX_workspace.r1;
	R1 = CTX_workspace.R1;
	r2 = CTX_workspace.r2;
	R2 = CTX_workspace.R2;
	r3 = CTX_workspace.r3;
	R3 = CTX_workspace.R3;
	r4 = CTX_workspace.r4;
	R4 = CTX_workspace.R4;
	S11ar = CTX_workspace.S11ar;
	S11br = CTX_workspace.S11br;
	S11cr = CTX_workspace.S11cr;
	S12ar = CTX_workspace.S12ar;
	S12br = CTX_workspace.S12br;
	S12cr = CTX_workspace.S12cr;
	S13ar = CTX_workspace.S13ar;
	S13br = CTX_workspace.S13br;
	S13cr = CTX_workspace.S13cr;
	S14ar = CTX_workspace.S14ar;
	S14br = CTX_workspace.S14br;
	S14cr = CTX_workspace.S14cr;
	S1a = CTX_workspace.S1a;
	S1b = CTX_workspace.S1b;
	S1c = CTX_workspace.S1c;
	S21a = CTX_workspace.S21a;
	S21an = CTX_workspace.S21an;
	S21b = CTX_workspace.S21b;
	S2a = CTX_workspace.S2a;
	S2an = CTX_workspace.S2an;
	S2b = CTX_workspace.S2b;
	s3 = CTX_workspace.s3;
	S31a = CTX_workspace.S31a;
	S31b = CTX_workspace.S31b;
	S31c = CTX_workspace.S31c;
	S32b = CTX_workspace.S32b;
	S32c = CTX_workspace.S32c;
	S3a = CTX_workspace.S3a;
	S3b = CTX_workspace.S3b;
	S3c = CTX_workspace.S3c;
	s4 = CTX_workspace.s4;
	S4 = CTX_workspace.S4;
	S5 = CTX_workspace.S5;
	S51 = CTX_workspace.S51;
	S52 = CTX_workspace.S52;
	S53 = CTX_workspace.S53;
	S54 = CTX_workspace.S54;
	S55 = CTX_workspace.S55;
	S56 = CTX_workspace.S56;
	S57 = CTX_workspace.S57;
	S58 = CTX_workspace.S58;
	S59 = CTX_workspace.S59;
	S61b = CTX_workspace.S61b;
	S61bn = CTX_workspace.S61bn;
	S6a = CTX_workspace.S6a;
	S6b = CTX_workspace.S6b;
	S6bn = CTX_workspace.S6bn;
	S7 = CTX_workspace.S7;
	S8 = CTX_workspace.S8;
	S81r = CTX_workspace.S81r;
	S82r = CTX_workspace.S82r;
	S83r = CTX_workspace.S83r;
	S9 = CTX_workspace.S9;
	S91 = CTX_workspace.S91;
	S92 = CTX_workspace.S92;
	S93 = CTX_workspace.S93;
	S94 = CTX_workspace.S94;
	S95 = CTX_workspace.S95;
	S96 = CTX_workspace.S96;
	S97 = CTX_workspace.S97;
	S98 = CTX_workspace.S98;
	S99 = CTX_workspace.S99;
	%SMC_pulse = CTX_workspace.SMC_pulse;
	syn_func_cor_d2 = CTX_workspace.syn_func_cor_d2;
	syn_func_cor_stn_a = CTX_workspace.syn_func_cor_stn_a;
	syn_func_cor_stn_n = CTX_workspace.syn_func_cor_stn_n;
	syn_func_gpe_gpe = CTX_workspace.syn_func_gpe_gpe;
	syn_func_gpe_gpi = CTX_workspace.syn_func_gpe_gpi;
	syn_func_gpe_stn = CTX_workspace.syn_func_gpe_stn;
	syn_func_gpi_th = CTX_workspace.syn_func_gpi_th;
	syn_func_stn_gpea = CTX_workspace.syn_func_stn_gpea;
	syn_func_stn_gpen = CTX_workspace.syn_func_stn_gpen;
	syn_func_stn_gpi = CTX_workspace.syn_func_stn_gpi;
	syn_func_str_dr = CTX_workspace.syn_func_str_dr;
	syn_func_str_indr = CTX_workspace.syn_func_str_indr;
	syn_func_th = CTX_workspace.syn_func_th;
	t = CTX_workspace.t;
	T = CTX_workspace.T;
	t_a = CTX_workspace.t_a;
	t_d_cor_d2 = CTX_workspace.t_d_cor_d2;
	t_d_cor_stn = CTX_workspace.t_d_cor_stn;
	t_d_d1_gpi = CTX_workspace.t_d_d1_gpi;
	t_d_d2_gpe = CTX_workspace.t_d_d2_gpe;
	t_d_gpe_gpe = CTX_workspace.t_d_gpe_gpe;
	t_d_gpe_gpi = CTX_workspace.t_d_gpe_gpi;
	t_d_gpe_stn = CTX_workspace.t_d_gpe_stn;
	t_d_gpi_th = CTX_workspace.t_d_gpi_th;
	t_d_stn_gpe = CTX_workspace.t_d_stn_gpe;
	t_d_stn_gpi = CTX_workspace.t_d_stn_gpi;
	t_d_th_cor = CTX_workspace.t_d_th_cor;
	t_list_cor = CTX_workspace.t_list_cor;
	t_list_gpe = CTX_workspace.t_list_gpe;
	t_list_gpi = CTX_workspace.t_list_gpi;
	t_list_stn = CTX_workspace.t_list_stn;
	t_list_str_dr = CTX_workspace.t_list_str_dr;
	t_list_str_indr = CTX_workspace.t_list_str_indr;
	t_list_th = CTX_workspace.t_list_th;
	t_vec = CTX_workspace.t_vec;
	ta2 = CTX_workspace.ta2;
	tau = CTX_workspace.tau;
	tau_i = CTX_workspace.tau_i;
	tauda = CTX_workspace.tauda;
	taudg = CTX_workspace.taudg;
	taudn = CTX_workspace.taudn;
	taudstngpea = CTX_workspace.taudstngpea;
	taudstngpen = CTX_workspace.taudstngpen;
	taura = CTX_workspace.taura;
	taurg = CTX_workspace.taurg;
	taurn = CTX_workspace.taurn;
	taurstngpea = CTX_workspace.taurstngpea;
	taurstngpen = CTX_workspace.taurstngpen;
	tb2 = CTX_workspace.tb2;
	tc2 = CTX_workspace.tc2;
	td1 = CTX_workspace.td1;
	td2 = CTX_workspace.td2;
	th1 = CTX_workspace.th1;
	th2 = CTX_workspace.th2;
	th3 = CTX_workspace.th3;
	th4 = CTX_workspace.th4;
    timespike = CTX_workspace.timespike;
	tm2 = CTX_workspace.tm2;
	tn2 = CTX_workspace.tn2;
	tn3 = CTX_workspace.tn3;
	tn4 = CTX_workspace.tn4;
	tp2 = CTX_workspace.tp2;
	tpeaka = CTX_workspace.tpeaka;
	tpeakg = CTX_workspace.tpeakg;
	tpeakn = CTX_workspace.tpeakn;
	tpeakstngpea = CTX_workspace.tpeakstngpea;
    tpeakstngpen = CTX_workspace.tpeakstngpen;
	tq2 = CTX_workspace.tq2;
	tr1 = CTX_workspace.tr1;
	tr2 = CTX_workspace.tr2;
	tr3 = CTX_workspace.tr3;
	tr4 = CTX_workspace.tr4;
	uce = CTX_workspace.uce;
	uci = CTX_workspace.uci;
	ue = CTX_workspace.ue;
	ui = CTX_workspace.ui;
	v1 = CTX_workspace.v1;
	V1 = CTX_workspace.V1;
	v2 = CTX_workspace.v2;
	V2 = CTX_workspace.V2;
	v3 = CTX_workspace.v3;
	V3 = CTX_workspace.V3;
	v4 = CTX_workspace.v4;
	V4 = CTX_workspace.V4;
	v5 = CTX_workspace.v5;
	V5 = CTX_workspace.V5;
	v6 = CTX_workspace.v6;
	V6 = CTX_workspace.V6;
	V7 = CTX_workspace.V7;
	V8 = CTX_workspace.V8;
	ve = CTX_workspace.ve;
	vge = CTX_workspace.vge;
	vgi = CTX_workspace.vgi;
	vi = CTX_workspace.vi;
	vsn = CTX_workspace.vsn;
	vstr_dr = CTX_workspace.vstr_dr;
	vstr_indr = CTX_workspace.vstr_indr;
	vth = CTX_workspace.vth;
	%vThal = CTX_workspace.vThal;
	Z = CTX_workspace.Z;
	Z1a = CTX_workspace.Z1a;
	z1adot = CTX_workspace.z1adot;
	Z1b = CTX_workspace.Z1b;

    
    pd = new_pd;
    corstim = new_corstim;
    tmax = new_tmax;
    dt = new_dt;
    n = new_n;
    Idbs_step = new_Idbs_step;
    Iappco_step = new_Iappco_step;
    len = new_len;

    %time step
    t=0:dt:len;
    ii = i+1; 
    window_size = windowlen; % ms
    crop_length = length(vth)-window_size/dt+1;
    % need to fix the ii and i variable as well
    ii = ii - crop_length + 1;

    %initial conditions (IC) to the different cell types
    % %%Setting initial matrices
    [SMC_pulse, ~] = createSMC(tmax,dt,14,0.2);
    %SMC_pulse = [SMC_pulse(:,crop_length:end) SMC_hold];
    %timespike_hold = timespike_hold+window_size*dt;
    %timespike = [timespike(timespike>(crop_length*dt)) timespike_hold];
    %timespike = timespike - crop_length*dt;
    
    vth= [vth(:,crop_length:end) zeros(n,length(t))]; %thalamic membrane voltage
    vsn= [vsn(:,crop_length:end) zeros(n,length(t))]; %STN membrane voltage
    vge= [vge(:,crop_length:end) zeros(n,length(t))]; %GPe membrane voltage
    vgi= [vgi(:,crop_length:end) zeros(n,length(t))]; %GPi membrane voltage
    vstr_indr= [vstr_indr(:,crop_length:end) zeros(n,length(t))]; %Indirect Striatum membrane voltage
    vstr_dr= [vstr_dr(:,crop_length:end) zeros(n,length(t))]; %Direct Striatum membrane voltage
    ve= [ve(:,crop_length:end) zeros(n,length(t))]; %Excitatory Cortex membrane voltage
    vi= [vi(:,crop_length:end) zeros(n,length(t))]; %Inhibitory Cortex membrane voltage
    ue=[ue(:,crop_length:end) zeros(n,length(t))];
    ui= [ui(:,crop_length:end) zeros(n,length(t))];
    Idbs= [Idbs(:,crop_length:end) Idbs_step];
    Iappco = [Iappco(1,crop_length:end) Iappco_step];
    len_t = length(t);
    Iappth=[Iappth(1,crop_length:end) SMC_pulse];

    for i=ii:ii+len_t-1 
        
        V1=vth(:,i-1);   
        V2=vsn(:,i-1);     
        V3=vge(:,i-1);    
        V4=vgi(:,i-1);
        V5=vstr_indr(:,i-1);
        V6=vstr_dr(:,i-1);
        V7=ve(:,i-1);
        V8=vi(:,i-1);

    % Synapse parameters 
    
    S21a(2:n)=S2a(1:n-1);
    S21a(1)=S2a(n);
    
    S21an(2:n)=S2an(1:n-1);
    S21an(1)=S2an(n);
    
    S21b(2:n)=S2b(1:n-1);
    S21b(1)=S2b(n);

    S31a(1:n-1)=S3a(2:n);
    S31a(n)=S3a(1);
    
    S31b(1:n-1)=S3b(2:n);
    S31b(n)=S3b(1);
    
    S31c(1:n-1)=S3c(2:n);
    S31c(n)=S3c(1);
    
    S32c(3:n)=S3c(1:n-2);
    S32c(1:2)=S3c(n-1:n);
    
    S32b(3:n)=S3b(1:n-2);
    S32b(1:2)=S3b(n-1:n);
    
    all = all;
    S11cr=S1c(all);
    S12cr=S1c(bll);
    S13cr=S1c(cll);
    S14cr=S1c(dll);

    S11br=S1b(ell);
    S12br=S1b(fll);
    S13br=S1b(gll);
    S14br=S1b(hll);

    S11ar=S1a(ill);
    S12ar=S1a(jll);
    S13ar=S1a(kll);
    S14ar=S1a(lll);

    S81r=S8(mll);
    S82r=S8(nll);
    S83r=S8(oll);

    S51(1:n-1)=S5(2:n);
    S51(n)=S5(1);
    S52(1:n-2)=S5(3:n);
    S52(n-1:n)=S5(1:2);
    S53(1:n-3)=S5(4:n);
    S53(n-2:n)=S5(1:3);
    S54(1:n-4)=S5(5:n);
    S54(n-3:n)=S5(1:4);
    S55(1:n-5)=S5(6:n);
    S55(n-4:n)=S5(1:5);
    S56(1:n-6)=S5(7:n);
    S56(n-5:n)=S5(1:6);
    S57(1:n-7)=S5(8:n);
    S57(n-6:n)=S5(1:7);
    S58(1:n-8)=S5(9:n);
    S58(n-7:n)=S5(1:8);
    S59(1:n-9)=S5(10:n);
    S59(n-8:n)=S5(1:9);

    S61b(1:n-1)=S6b(2:n);
    S61b(n)=S6b(1);
    
    S61bn(1:n-1)=S6bn(2:n);
    S61bn(n)=S6bn(1);
    
    S91(1:n-1)=S9(2:n);
    S91(n)=S9(1);
    S92(1:n-2)=S9(3:n);
    S92(n-1:n)=S9(1:2);
    S93(1:n-3)=S9(4:n);
    S93(n-2:n)=S9(1:3);
    S94(1:n-4)=S9(5:n);
    S94(n-3:n)=S9(1:4);
    S95(1:n-5)=S9(6:n);
    S95(n-4:n)=S9(1:5);
    S96(1:n-6)=S9(7:n);
    S96(n-5:n)=S9(1:6);
    S97(1:n-7)=S9(8:n);
    S97(n-6:n)=S9(1:7);
    S98(1:n-8)=S9(9:n);
    S98(n-7:n)=S9(1:8);
    S99(1:n-9)=S9(10:n);
    S99(n-8:n)=S9(1:9);
    
    m1=th_minf(V1);
    m3=gpe_minf(V3);m4=gpe_minf(V4);
    n3=gpe_ninf(V3);n4=gpe_ninf(V4);
    h1=th_hinf(V1);
    h3=gpe_hinf(V3);h4=gpe_hinf(V4);
    p1=th_pinf(V1);
    a3=gpe_ainf(V3);a4=gpe_ainf(V4);
    s3=gpe_sinf(V3);s4=gpe_sinf(V4);
    r1=th_rinf(V1);
    r3=gpe_rinf(V3);r4=gpe_rinf(V4);

    tn3=gpe_taun(V3);tn4=gpe_taun(V4);
    th1=th_tauh(V1);
    th3=gpe_tauh(V3);th4=gpe_tauh(V4);
    tr1=th_taur(V1);tr3=30;tr4=30;
    
    n2=stn_ninf(V2);
    m2=stn_minf(V2);
    h2=stn_hinf(V2);
    a2=stn_ainf(V2);
    b2=stn_binf(V2);
    c2=stn_cinf(V2);
    d2=stn_d2inf(V2);
    d1=stn_d1inf(V2);
    p2=stn_pinf(V2);
    q2=stn_qinf(V2);
    r2=stn_rinf(V2);
 
    td2=130;
    tr2=2;
    tn2=stn_taun(V2);
    tm2=stn_taum(V2);
    th2=stn_tauh(V2);
    ta2=stn_taua(V2);
    tb2=stn_taub(V2);
    tc2=stn_tauc(V2);
    td1=stn_taud1(V2);
    tp2=stn_taup(V2);
    tq2=stn_tauq(V2);

    Ecasn=con*log(Cao./CAsn2);
   
   
    %thalamic cell currents
    Il1=gl(1)*(V1-El(1));
    Ina1=gna(1)*(m1.^3).*H1.*(V1-Ena(1));
    Ik1=gk(1)*((0.75*(1-H1)).^4).*(V1-Ek(1));
    gt = gt;
    It1=gt(1)*(p1.^2).*R1.*(V1-Et);
    Igith=ggith*(V1-Esyn(6)).*(S4); 
    
 

    %STN cell currents
    Ina2=gna(2)*(M2.^3).*H2.*(V2-Ena(2));
    Ik2=gk(2)*(N2.^4).*(V2-Ek(2));
    Ia2=ga1*(A2.^2).*(B2).*(V2-Ek(2));
    IL2=gL*(C2.^2).*(D1).*(D2).*(V2-Ecasn);
    It2=(gt(2)*(P2.^2).*(Q2).*(V2-Ecasn));
    Icak2=gcak*(R2.^2).*(V2-Ek(2));
    Il2=gl(2)*(V2-El(2));
    Igesn=(ggesn*((V2-Esyn(1)).*(S3a+S31a))); 
    Icorsnampa=gcorsna.*(V2-Esyn(2)).*(S6b+S61b);
    Icorsnnmda=gcorsnn.*(V2-Esyn(2)).*(S6bn+S61bn);

    %GPe cell currents
    Il3=gl(3)*(V3-El(3));
    Ik3=gk(3)*(N3.^4).*(V3-Ek(3));
    Ina3=gna(3)*(m3.^3).*H3.*(V3-Ena(3));
    It3=gt(3)*(a3.^3).*R3.*(V3-Eca(3));
    gca = gca;
    Ica3=gca(3)*(s3.^2).*(V3-Eca(3));
    Iahp3=gahp(3)*(V3-Ek(3)).*(CA3./(CA3+k1(3)));
    Isngeampa=(gsngea).*((V3-Esyn(2)).*(S2a+S21a)); 
    Isngenmda=(gsngen).*((V3-Esyn(2)).*(S2an+S21an)); 
    Igege=(0.25*(pd*3+1))*(ggege).*((V3-Esyn(3)).*(S31c+S32c)); 
    Istrgpe=gstrgpe*(V3-Esyn(6)).*(S5+S51+S52+S53+S54+S55+S56+S57+S58+S59);
    Iappgpe=3-2*corstim*~pd; %Modulation only during cortical stim to maintain mean firing rate

    %GPi cell currents
    Il4=gl(3)*(V4-El(3));
    Ik4=gk(3)*(N4.^4).*(V4-Ek(3));
    Ina4=gna(3)*(m4.^3).*H4.*(V4-Ena(3));
    It4=gt(3)*(a4.^3).*R4.*(V4-Eca(3));
    Ica4=gca(3)*(s4.^2).*(V4-Eca(3));
    Iahp4=gahp(3)*(V4-Ek(3)).*(CA4./(CA4+k1(3)));
    Isngi=(gsngi).*((V4-Esyn(4)).*(S2b+S21b));
    Igigi=ggigi*((V4-Esyn(5)).*(S31b+S32b)); 
    Istrgpi=gstrgpi*(V4-Esyn(6)).*(S9+S91+S92+S93+S94+S95+S96+S97+S98+S99);
    Iappgpi=3;

    %Striatum D2 cell currents
    Ina5=gna(4)*(m5.^3).*h5.*(V5-Ena(4));
    Ik5=gk(4)*(n5.^4).*(V5-Ek(4));
    Il5=gl(4)*(V5-El(4));
    Im5=(2.6-1.1*pd)*gm*p5.*(V5-Em);
    Igaba5=(ggaba/4)*(V5-Esyn(7)).*(S11cr+S12cr+S13cr+S14cr);
    Icorstr5=gcorindrstr*(V5-Esyn(2)).*(S6a);
    
    %Striatum D1 cell currents
    Ina6=gna(4)*(m6.^3).*h6.*(V6-Ena(4));
    Ik6=gk(4)*(n6.^4).*(V6-Ek(4));
    Il6=gl(4)*(V6-El(4));
    Im6=(2.6-1.1*pd)*gm*p6.*(V6-Em);
    Igaba6=(ggaba/3)*(V6-Esyn(7)).*(S81r+S82r+S83r);
    Icorstr6=gcordrstr.*(V6-Esyn(2)).*(S6a);
    
    %Excitatory Neuron Currents
    Iie=gie*(V7-Esyn(1)).*(S11br+S12br+S13br+S14br);
    Ithcor=gthcor*(V7-Esyn(2)).*(S7);

    
    %Inhibitory Neuron Currents
    Iei=gei*(V8-Esyn(2)).*(S11ar+S12ar+S13ar+S14ar);

    %Differential Equations for cells
    %thalamic
    vth(:,i)= V1+dt*(1/Cm*(-Il1-Ik1-Ina1-It1-Igith+Iappth(i)));
    H1=H1+dt*((h1-H1)./th1);
    R1=R1+dt*((r1-R1)./tr1);

for j=1:n

if (vth(j,i-1)<-10 && vth(j,i)>-10) % check for input spike
     t_list_th(j).times = [t_list_th(j).times; 1];
end   
   % Calculate synaptic current due to current and past input spikes
   S7(j) = sum(syn_func_th(t_list_th(j).times));

   % Update spike times
   if t_list_th(j).times
     t_list_th(j).times = t_list_th(j).times + 1;
     if (t_list_th(j).times(1) == t_a/dt)  % Reached max duration of syn conductance
       t_list_th(j).times = t_list_th(j).times((2:max(size(t_list_th(j).times))));
     end
   end
end

    vsn(:,i)=V2+dt*(1/Cm*(-Ina2-Ik2-Ia2-IL2-It2-Icak2-Il2-Igesn-Icorsnampa-Icorsnnmda+Idbs(i))); %STN-DBS
    N2=N2+dt*((n2-N2)./tn2); 
    H2=H2+dt*((h2-H2)./th2);
    M2=M2+dt*((m2-M2)./tm2); 
    A2=A2+dt*((a2-A2)./ta2);
    B2=B2+dt*((b2-B2)./tb2); 
    C2=C2+dt*((c2-C2)./tc2);
    D2=D2+dt*((d2-D2)./td2); 
    D1=D1+dt*((d1-D1)./td1);
    P2=P2+dt*((p2-P2)./tp2); 
    Q2=Q2+dt*((q2-Q2)./tq2);
    R2=R2+dt*((r2-R2)./tr2); 
    
    CAsn2=CAsn2+dt*((-alp*(IL2+It2))-(Kca*CAsn2));
for j=1:n

if (vsn(j,i-1)<-10 && vsn(j,i)>-10) % check for input spike
     t_list_stn(j).times = [t_list_stn(j).times; 1];
end   
   % Calculate synaptic current due to current and past input spikes
   S2a(j) = sum(syn_func_stn_gpea(t_list_stn(j).times));
   S2an(j) = sum(syn_func_stn_gpen(t_list_stn(j).times));

   S2b(j) = sum(syn_func_stn_gpi(t_list_stn(j).times));

   % Update spike times
   if t_list_stn(j).times
     t_list_stn(j).times = t_list_stn(j).times + 1;
     if (t_list_stn(j).times(1) == t_a/dt)  % Reached max duration of syn conductance
       t_list_stn(j).times = t_list_stn(j).times((2:max(size(t_list_stn(j).times))));
     end
   end
end
    
    %GPe 
    vge(:,i)=V3+dt*(1/Cm*(-Il3-Ik3-Ina3-It3-Ica3-Iahp3-Isngeampa-Isngenmda-Igege-Istrgpe+Iappgpe));
    N3=N3+dt*(0.1*(n3-N3)./tn3);
    H3=H3+dt*(0.05*(h3-H3)./th3);
    R3=R3+dt*(1*(r3-R3)./tr3);
    CA3=CA3+dt*(1*10^-4*(-Ica3-It3-kca(3)*CA3));
for j=1:n

if (vge(j,i-1)<-10 && vge(j,i)>-10) % check for input spike
     t_list_gpe(j).times = [t_list_gpe(j).times; 1];
end   
   % Calculate synaptic current due to current and past input spikes
   S3a(j) = sum(syn_func_gpe_stn(t_list_gpe(j).times));
   S3b(j) = sum(syn_func_gpe_gpi(t_list_gpe(j).times));
   S3c(j) = sum(syn_func_gpe_gpe(t_list_gpe(j).times));


   % Update spike times
   if t_list_gpe(j).times
     t_list_gpe(j).times = t_list_gpe(j).times + 1;
     if (t_list_gpe(j).times(1) == t_a/dt)  % Reached max duration of syn conductance
       t_list_gpe(j).times = t_list_gpe(j).times((2:max(size(t_list_gpe(j).times))));
     end
   end
end
    
    %GPi  
%     vgi(:,i)=V4+dt*(1/Cm*(-Il4-Ik4-Ina4-It4-Ica4-Iahp4-Isngi-Igigi-Istrgpi+Iappgpi+Idbs(i))); %% GPi-DBS
    vgi(:,i)=V4+dt*(1/Cm*(-Il4-Ik4-Ina4-It4-Ica4-Iahp4-Isngi-Igigi-Istrgpi+Iappgpi));
    N4=N4+dt*(0.1*(n4-N4)./tn4);
    H4=H4+dt*(0.05*(h4-H4)./th4);
    R4=R4+dt*(1*(r4-R4)./tr4);
    CA4=CA4+dt*(1*10^-4*(-Ica4-It4-kca(3)*CA4));

for j=1:n

if (vgi(j,i-1)<-10 && vgi(j,i)>-10) % check for input spike
     t_list_gpi(j).times = [t_list_gpi(j).times; 1];
end   
   % Calculate synaptic current due to current and past input spikes
   S4(j) = sum(syn_func_gpi_th(t_list_gpi(j).times));


   % Update spike times
   if t_list_gpi(j).times
     t_list_gpi(j).times = t_list_gpi(j).times + 1;
     if (t_list_gpi(j).times(1) == t_a/dt)  % Reached max duration of syn conductance
       t_list_gpi(j).times = t_list_gpi(j).times((2:max(size(t_list_gpi(j).times))));
     end
   end
end
    
    %Striatum D2
 vstr_indr(:,i)=V5+(dt/Cm)*(-Ina5-Ik5-Il5-Im5-Igaba5-Icorstr5);
 m5=m5+dt*(alpham(V5).*(1-m5)-betam(V5).*m5);
 h5=h5+dt*(alphah(V5).*(1-h5)-betah(V5).*h5);
 n5=n5+dt*(alphan(V5).*(1-n5)-betan(V5).*n5);
 p5=p5+dt*(alphap(V5).*(1-p5)-betap(V5).*p5);
 S1c=S1c+dt*((Ggaba(V5).*(1-S1c))-(S1c/tau_i));

for j=1:n

if (vstr_indr(j,i-1)<-10 && vstr_indr(j,i)>-10) % check for input spike
     t_list_str_indr(j).times = [t_list_str_indr(j).times; 1];
end   
   % Calculate synaptic current due to current and past input spikes
   S5(j) = sum(syn_func_str_indr(t_list_str_indr(j).times));

   % Update spike times
   if t_list_str_indr(j).times
     t_list_str_indr(j).times = t_list_str_indr(j).times + 1;
     if (t_list_str_indr(j).times(1) == t_a/dt)  % Reached max duration of syn conductance
       t_list_str_indr(j).times = t_list_str_indr(j).times((2:max(size(t_list_str_indr(j).times))));
     end
   end
end

% %Striatum D1 type
 vstr_dr(:,i)=V6+(dt/Cm)*(-Ina6-Ik6-Il6-Im6-Igaba6-Icorstr6);
 m6=m6+dt*(alpham(V6).*(1-m6)-betam(V6).*m6);
 h6=h6+dt*(alphah(V6).*(1-h6)-betah(V6).*h6);
 n6=n6+dt*(alphan(V6).*(1-n6)-betan(V6).*n6);
 p6=p6+dt*(alphap(V6).*(1-p6)-betap(V6).*p6);
 S8=S8+dt*((Ggaba(V6).*(1-S8))-(S8/tau_i));

 
 for j=1:n

if (vstr_dr(j,i-1)<-10 && vstr_dr(j,i)>-10) % check for input spike
     t_list_str_dr(j).times = [t_list_str_dr(j).times; 1];
end   
   % Calculate synaptic current due to current and past input spikes
   S9(j) = sum(syn_func_str_dr(t_list_str_dr(j).times));

   % Update spike times
   if t_list_str_dr(j).times
     t_list_str_dr(j).times = t_list_str_dr(j).times + 1;
     if (t_list_str_dr(j).times(1) == t_a/dt)  % Reached max duration of syn conductance
       t_list_str_dr(j).times = t_list_str_dr(j).times((2:max(size(t_list_str_dr(j).times))));
     end
   end
 end

%Excitatory Neuron
    ve(:,i)=V7+dt*((0.04*(V7.^2))+(5*V7)+140-ue(:,i-1)-Iie-Ithcor+Iappco(i));
    ue(:,i)=ue(:,i-1)+dt*(ae*((be*V7)-ue(:,i-1)));
    
   for j=1:n
        if ve(j,i-1)>=30
        ve(j,i)=ce;
        ue(j,i)=ue(j,i-1)+de;
        
 t_list_cor(j).times = [t_list_cor(j).times; 1];
        end
   
   % Calculate synaptic current due to current and past input spikes
   S6a(j) = sum(syn_func_cor_d2(t_list_cor(j).times));
   S6b(j) = sum(syn_func_cor_stn_a(t_list_cor(j).times));
   S6bn(j) = sum(syn_func_cor_stn_n(t_list_cor(j).times));

   % Update spike times
   if t_list_cor(j).times
     t_list_cor(j).times = t_list_cor(j).times + 1;
     if (t_list_cor(j).times(1) == t_a/dt)  % Reached max duration of syn conductance
       t_list_cor(j).times = t_list_cor(j).times((2:max(size(t_list_cor(j).times))));
     end
   end
   
   end        
    
    ace=find(ve(:,i-1)<-10 & ve(:,i)>-10);
    uce=zeros(n,1); uce(ace)=gpeak/(tau*exp(-1))/dt;
    S1a=S1a+dt*Z1a; 
    z1adot=uce-2/tau*Z1a-1/(tau^2)*S1a;
    Z1a=Z1a+dt*z1adot;
    
    %Inhibitory InterNeuron
    vi(:,i)=V8+dt*((0.04*(V8.^2))+(5*V8)+140-ui(:,i-1)-Iei+Iappco(i));
    ui(:,i)=ui(:,i-1)+dt*(ai*((bi*V8)-ui(:,i-1)));
    
   for j=1:n
        if vi(j,i-1)>=30
        vi(j,i)=ci;
        ui(j,i)=ui(j,i-1)+di;
        end
   end
        
    
    aci=find(vi(:,i-1)<-10 & vi(:,i)>-10);
    uci=zeros(n,1); uci(aci)=gpeak/(tau*exp(-1))/dt;
    S1b=S1b+dt*Z1b; 
    z1bdot=uci-2/tau*Z1b-1/(tau^2)*S1b;
    Z1b=Z1b+dt*z1bdot;

    end

    % START FIGURE CREATION
%     figure;
%     plot(vgi(1,:))
%     title('V_g_p_i, 0.02 ms dt')
%     xlabel('Step')
%     ylabel('Voltage (mV)')
% 
%     figure;
%     plot(vgi(1,:))
%     title('V_t_h, 0.02 ms dt')
%     hold on;
%     plot(-10.*Iappth)
%     xlabel('Step')
%     ylabel('Voltage (mV)')
    % END FIGURE CREATION
    
    [vgpi] = vgi;
    [vThal] = vth;
    [SMC_pulse] = Iappth;
    % create Struct
    CTX_workspace.a2 = a2;
    CTX_workspace.A2 = A2;
    CTX_workspace.a3 = a3;
    CTX_workspace.a4 = a4;
    CTX_workspace.ace = ace;
    CTX_workspace.aci = aci;
    CTX_workspace.ae = ae;
    CTX_workspace.ai = ai;
    CTX_workspace.all = all;
    CTX_workspace.alp = alp;
    CTX_workspace.b2 = b2;
    CTX_workspace.B2 = B2;
    CTX_workspace.be = be;
    CTX_workspace.bi = bi;
    CTX_workspace.bll = bll;
    CTX_workspace.c2 = c2;
    CTX_workspace.C2 = C2;
    CTX_workspace.CA2 = CA2;
    CTX_workspace.CA3 = CA3;
    CTX_workspace.CA4 = CA4;
    CTX_workspace.Cao = Cao;
    CTX_workspace.CAsn2 = CAsn2;
    CTX_workspace.ce = ce;
    CTX_workspace.ci = ci;
    CTX_workspace.cll = cll;
    CTX_workspace.Cm = Cm;
    CTX_workspace.con = con;
    CTX_workspace.const = const;
    CTX_workspace.const1 = const1;
    CTX_workspace.const2 = const2;
    CTX_workspace.corstim = corstim;
    CTX_workspace.d1 = d1;
    CTX_workspace.D1 = D1;
    CTX_workspace.d2 = d2;
    CTX_workspace.D2 = D2;
    CTX_workspace.de = de;
    CTX_workspace.di = di;
    CTX_workspace.dll = dll;
    CTX_workspace.dt = dt;
    CTX_workspace.Eca = Eca;
    CTX_workspace.Ecasn = Ecasn;
    CTX_workspace.Ek = Ek;
    CTX_workspace.El = El;
    CTX_workspace.ell = ell;
    CTX_workspace.Em = Em;
    CTX_workspace.Ena = Ena;
    CTX_workspace.Esyn = Esyn;
    CTX_workspace.Et = Et;
    CTX_workspace.F = F;
    CTX_workspace.fa = fa;
    CTX_workspace.fg = fg;
    CTX_workspace.fll = fll;
    CTX_workspace.fn = fn;
    CTX_workspace.fstngpea = fstngpea;
    CTX_workspace.fstngpen = fstngpen;
    CTX_workspace.ga1 = ga1;
    CTX_workspace.gahp = gahp;
    CTX_workspace.gca = gca;
    CTX_workspace.gcak = gcak;
    CTX_workspace.gcordrstr = gcordrstr;
    CTX_workspace.gcorindrstr = gcorindrstr;
    CTX_workspace.gcorsna = gcorsna;
    CTX_workspace.gcorsnn = gcorsnn;
    CTX_workspace.gei = gei;
    CTX_workspace.ggaba = ggaba;
    CTX_workspace.ggege = ggege;
    CTX_workspace.ggesn = ggesn;
    CTX_workspace.ggigi = ggigi;
    CTX_workspace.ggith = ggith;
    CTX_workspace.gie = gie;
    CTX_workspace.gk = gk;
    CTX_workspace.gl = gl;
    CTX_workspace.gL = gL;
    CTX_workspace.gll = gll;
    CTX_workspace.gm = gm;
    CTX_workspace.gna = gna;
    CTX_workspace.gpeak = gpeak;
    CTX_workspace.gpeak1 = gpeak1;
    CTX_workspace.gsngea = gsngea;
    CTX_workspace.gsngen = gsngen;
    CTX_workspace.gsngi = gsngi;
    CTX_workspace.gstrgpe = gstrgpe;
    CTX_workspace.gstrgpi = gstrgpi;
    CTX_workspace.gt = gt;
    CTX_workspace.gthcor = gthcor;
    CTX_workspace.h1 = h1;
    CTX_workspace.H1 = H1;
    CTX_workspace.h2 = h2;
    CTX_workspace.H2 = H2;
    CTX_workspace.h3 = h3;
    CTX_workspace.H3 = H3;
    CTX_workspace.h4 = h4;
    CTX_workspace.H4 = h4;
    CTX_workspace.h5 = h5;
    CTX_workspace.h6 = h6;
    CTX_workspace.hll = hll;
    CTX_workspace.i = i;
    CTX_workspace.Ia2 = Ia2;
    CTX_workspace.Iahp3 = Iahp3;
    CTX_workspace.Iahp4 = Iahp4;
    CTX_workspace.Iappco = Iappco;
    CTX_workspace.Iappgpe = Iappgpe;
    CTX_workspace.Iappgpi = Iappgpi;
    CTX_workspace.Iappth = Iappth;
    CTX_workspace.Ica3 = Ica3;
    CTX_workspace.Ica4 = Ica4;
    CTX_workspace.Icak2 = Icak2;
    CTX_workspace.Icorsnampa = Icorsnampa;
    CTX_workspace.Icorsnnmda = Icorsnnmda;
    CTX_workspace.Icorstr5 = Icorstr5;
    CTX_workspace.Icorstr6 = Icorstr6;
    CTX_workspace.Idbs = Idbs;
    CTX_workspace.Iei = Iei;
    CTX_workspace.Igaba5 = Igaba5;
    CTX_workspace.Igaba6 = Igaba6;
    CTX_workspace.Igege = Igege;
    CTX_workspace.Igesn = Igesn;
    CTX_workspace.Igigi = Igigi;
    CTX_workspace.Igith = Igith;
    CTX_workspace.Iie = Iie;
    CTX_workspace.Ik1 = Ik1;
    CTX_workspace.Ik2 = Ik2;
    CTX_workspace.Ik3 = Ik3;
    CTX_workspace.Ik4 = Ik4;
    CTX_workspace.Ik5 = Ik5;
    CTX_workspace.Ik6 = Ik6;
    CTX_workspace.Il1 = Il1;
    CTX_workspace.Il2 = Il2;
    CTX_workspace.IL2 = IL2;
	CTX_workspace.Il3 = Il3;
	CTX_workspace.Il4 = Il4;
	CTX_workspace.Il5 = Il5;
	CTX_workspace.Il6 = Il6;
	CTX_workspace.ill = ill;
	CTX_workspace.Im5 = Im5;
	CTX_workspace.Im6 = Im6;
	CTX_workspace.Ina1 = Ina1;
	CTX_workspace.Ina2 = Ina2;
	CTX_workspace.Ina3 = Ina3;
	CTX_workspace.Ina4 = Ina4;
	CTX_workspace.Ina5 = Ina5;
	CTX_workspace.Ina6 = Ina6;
	CTX_workspace.Isngeampa = Isngeampa;
	CTX_workspace.Isngenmda = Isngenmda;
	CTX_workspace.Isngi = Isngi;
	CTX_workspace.Istrgpe = Istrgpe;
	CTX_workspace.Istrgpi = Istrgpi;
	CTX_workspace.It1 = It1;
	CTX_workspace.It2 = It2;
	CTX_workspace.It3 = It3;
	CTX_workspace.It4 = It4;
	CTX_workspace.Ithcor = Ithcor;
	CTX_workspace.j = j;
	CTX_workspace.jll = jll;
	CTX_workspace.k1 = k1;
	CTX_workspace.kca = kca;
	CTX_workspace.Kca = Kca;
	CTX_workspace.kll = kll;
	CTX_workspace.lll = lll;
	CTX_workspace.m1 = m1;
	CTX_workspace.m2 = m2;
	CTX_workspace.M2 = M2;
	CTX_workspace.m3 = m3;
	CTX_workspace.m4 = m4;
	CTX_workspace.m5 = m5;
	CTX_workspace.m6 = m6;
	CTX_workspace.mll = mll;
	CTX_workspace.n = n;
	CTX_workspace.n2 = n2;
	CTX_workspace.N2 = N2;
	CTX_workspace.n3 = n3;
	CTX_workspace.N3 = N3;
	CTX_workspace.n4 = n4;
	CTX_workspace.N4 = N4;
	CTX_workspace.n5 = n5;
	CTX_workspace.n6 = n6;
	CTX_workspace.nll = nll;
	CTX_workspace.oll = oll;
	CTX_workspace.p1 = p1;
	CTX_workspace.p2 = p2;
	CTX_workspace.P2 = P2;
	CTX_workspace.p5 = p5;
	CTX_workspace.p6 = p6;
	CTX_workspace.pd = pd;
	CTX_workspace.q2 = q2;
	CTX_workspace.Q2 = Q2;
	CTX_workspace.R = R;
	CTX_workspace.r1 = r1;
	CTX_workspace.R1 = R1;
	CTX_workspace.r2 = r2;
	CTX_workspace.R2 = R2;
	CTX_workspace.r3 = r3;
	CTX_workspace.R3 = R3;
	CTX_workspace.r4 = r4;
	CTX_workspace.R4 = R4;
	CTX_workspace.S11ar = S11ar;
	CTX_workspace.S11br = S11br;
	CTX_workspace.S11cr = S11cr;
	CTX_workspace.S12ar = S12ar;
	CTX_workspace.S12br = S12br;
	CTX_workspace.S12cr = S12cr;
	CTX_workspace.S13ar = S13ar;
	CTX_workspace.S13br = S13br;
	CTX_workspace.S13cr = S13cr;
	CTX_workspace.S14ar = S14ar;
	CTX_workspace.S14br = S14br;
	CTX_workspace.S14cr = S14cr;
	CTX_workspace.S1a = S1a;
	CTX_workspace.S1b = S1b;
	CTX_workspace.S1c = S1c;
	CTX_workspace.S21a = S21a;
	CTX_workspace.S21an = S21an;
	CTX_workspace.S21b = S21b;
	CTX_workspace.S2a = S2a;
	CTX_workspace.S2an = S2an;
	CTX_workspace.S2b = S2b;
	CTX_workspace.s3 = s3;
	CTX_workspace.S31a = S31a;
	CTX_workspace.S31b = S31b;
	CTX_workspace.S31c = S31c;
	CTX_workspace.S32b = S32b;
	CTX_workspace.S32c = S32c;
	CTX_workspace.S3a = S3a;
	CTX_workspace.S3b = S3b;
	CTX_workspace.S3c = S3c;
	CTX_workspace.s4 = s4;
	CTX_workspace.S4 = S4;
	CTX_workspace.S5 = S5;
	CTX_workspace.S51 = S51;
	CTX_workspace.S52 = S52;
	CTX_workspace.S53 = S53;
	CTX_workspace.S54 = S54;
	CTX_workspace.S55 = S55;
	CTX_workspace.S56 = S56;
	CTX_workspace.S57 = S57;
	CTX_workspace.S58 = S58;
	CTX_workspace.S59 = S59;
	CTX_workspace.S61b = S61b;
	CTX_workspace.S61bn = S61bn;
	CTX_workspace.S6a = S6a;
	CTX_workspace.S6b = S6b;
	CTX_workspace.S6bn = S6bn;
	CTX_workspace.S7 = S7;
	CTX_workspace.S8 = S8;
	CTX_workspace.S81r = S81r;
	CTX_workspace.S82r = S82r;
	CTX_workspace.S83r = S83r;
	CTX_workspace.S9 = S9;
	CTX_workspace.S91 = S91;
	CTX_workspace.S92 = S92;
	CTX_workspace.S93 = S93;
	CTX_workspace.S94 = S94;
	CTX_workspace.S95 = S95;
	CTX_workspace.S96 = S96;
	CTX_workspace.S97 = S97;
	CTX_workspace.S98 = S98;
	CTX_workspace.S99 = S99;
	CTX_workspace.SMC_pulse = SMC_pulse;
	CTX_workspace.syn_func_cor_d2 = syn_func_cor_d2;
	CTX_workspace.syn_func_cor_stn_a = syn_func_cor_stn_a;
	CTX_workspace.syn_func_cor_stn_n = syn_func_cor_stn_n;
	CTX_workspace.syn_func_gpe_gpe = syn_func_gpe_gpe;
	CTX_workspace.syn_func_gpe_gpi = syn_func_gpe_gpi;
	CTX_workspace.syn_func_gpe_stn = syn_func_gpe_stn;
	CTX_workspace.syn_func_gpi_th = syn_func_gpi_th;
	CTX_workspace.syn_func_stn_gpea = syn_func_stn_gpea;
	CTX_workspace.syn_func_stn_gpen = syn_func_stn_gpen;
	CTX_workspace.syn_func_stn_gpi = syn_func_stn_gpi;
	CTX_workspace.syn_func_str_dr = syn_func_str_dr;
	CTX_workspace.syn_func_str_indr = syn_func_str_indr;
	CTX_workspace.syn_func_th = syn_func_th;
	CTX_workspace.t = t;
	CTX_workspace.T = T;
	CTX_workspace.t_a = t_a;
	CTX_workspace.t_d_cor_d2 = t_d_cor_d2;
	CTX_workspace.t_d_cor_stn = t_d_cor_stn;
	CTX_workspace.t_d_d1_gpi = t_d_d1_gpi;
	CTX_workspace.t_d_d2_gpe = t_d_d2_gpe;
	CTX_workspace.t_d_gpe_gpe = t_d_gpe_gpe;
	CTX_workspace.t_d_gpe_gpi = t_d_gpe_gpi;
	CTX_workspace.t_d_gpe_stn = t_d_gpe_stn;
	CTX_workspace.t_d_gpi_th = t_d_gpi_th;
	CTX_workspace.t_d_stn_gpe = t_d_stn_gpe;
	CTX_workspace.t_d_stn_gpi = t_d_stn_gpi;
	CTX_workspace.t_d_th_cor = t_d_th_cor;
	CTX_workspace.t_list_cor = t_list_cor;
	CTX_workspace.t_list_gpe = t_list_gpe;
	CTX_workspace.t_list_gpi = t_list_gpi;
	CTX_workspace.t_list_stn = t_list_stn;
	CTX_workspace.t_list_str_dr = t_list_str_dr;
	CTX_workspace.t_list_str_indr = t_list_str_indr;
	CTX_workspace.t_list_th = t_list_th;
	CTX_workspace.t_vec = t_vec;
	CTX_workspace.ta2 = ta2;
	CTX_workspace.tau = tau;
	CTX_workspace.tau_i = tau_i;
	CTX_workspace.tauda = tauda;
	CTX_workspace.taudg = taudg;
	CTX_workspace.taudn = taudn;
	CTX_workspace.taudstngpea = taudstngpea;
	CTX_workspace.taudstngpen = taudstngpen;
	CTX_workspace.taura = taura;
	CTX_workspace.taurg = taurg;
	CTX_workspace.taurn = taurn;
	CTX_workspace.taurstngpea = taurstngpea;
	CTX_workspace.taurstngpen = taurstngpen;
	CTX_workspace.tb2 = tb2;
	CTX_workspace.tc2 = tc2;
	CTX_workspace.td1 = td1;
	CTX_workspace.td2 = td2;
	CTX_workspace.th1 = th1;
	CTX_workspace.th2 = th2;
	CTX_workspace.th3 = th3;
	CTX_workspace.th4 = th4;
    CTX_workspace.timespike = timespike;
	CTX_workspace.tm2 = tm2;
	CTX_workspace.tmax = tmax;
	CTX_workspace.tn2 = tn2;
	CTX_workspace.tn3 = tn3;
	CTX_workspace.tn4 = tn4;
	CTX_workspace.tp2 = tp2;
	CTX_workspace.tpeaka = tpeaka;
	CTX_workspace.tpeakg = tpeakg;
	CTX_workspace.tpeakn = tpeakn;
	CTX_workspace.tpeakstngpea = tpeakstngpea;
    CTX_workspace.tpeakstngpen = tpeakstngpen;
	CTX_workspace.tq2 = tq2;
	CTX_workspace.tr1 = tr1;
	CTX_workspace.tr2 = tr2;
	CTX_workspace.tr3 = tr3;
	CTX_workspace.tr4 = tr4;
	CTX_workspace.uce = uce;
	CTX_workspace.uci = uci;
	CTX_workspace.ue = ue;
	CTX_workspace.ui = ui;
	CTX_workspace.v1 = v1;
	CTX_workspace.V1 = V1;
	CTX_workspace.v2 = v2;
	CTX_workspace.V2 = V2;
	CTX_workspace.v3 = v3;
	CTX_workspace.V3 = V3;
	CTX_workspace.v4 = v4;
	CTX_workspace.V4 = V4;
	CTX_workspace.v5 = v5;
	CTX_workspace.V5 = V5;
	CTX_workspace.v6 = v6;
	CTX_workspace.V6 = V6;
	CTX_workspace.V7 = V7;
	CTX_workspace.V8 = V8;
	CTX_workspace.ve = ve;
	CTX_workspace.vge = vge;
	CTX_workspace.vgi = vgi;
	CTX_workspace.vi = vi;
	CTX_workspace.vsn = vsn;
	CTX_workspace.vstr_dr = vstr_dr;
	CTX_workspace.vstr_indr = vstr_indr;
	CTX_workspace.vth = vth;
	CTX_workspace.vThal = vThal;
	CTX_workspace.Z = Z;
	CTX_workspace.Z1a = Z1a;
	CTX_workspace.z1adot = z1adot;
	CTX_workspace.Z1b = Z1b;
    save("CTX_workspace.mat","CTX_workspace");

    clear CTX_workspace
    CTX_workspace.timespike = [];

%     vTotal.vth = vth(:,end);
%     vTotal.vsn = vsn(:,end);
%     vTotal.vge = vge(:,end);
%     vTotal.vgi = vgi(:,end);
%     vTotal.vstr_indr = vstr_indr(:,end);
%     vTotal.vstr_dr = vstr_dr(:,end);

%     [TH_APs]  = find_spike_times(vth,t,n);
%     [STN_APs] = find_spike_times(vsn,t,n);
%     [GPe_APs] = find_spike_times(vge,t,n);
%     [GPi_APs] = find_spike_times(vgi,t,n);
%     [Striat_APs_indr]=find_spike_times(vstr_indr,t,n);
%     [Striat_APs_dr]=find_spike_times(vstr_dr,t,n);
%     [Cor_APs] = find_spike_times([ve;vi],t,2*n);
end

function [beta_vec, error_index] = sliding_window_2(v,vth,SMC,params,n,tmax,timespike,dt,stride,windowlen)
%Used to calculate the beta and EI based on window size and stride determined from data    
    
    % set up number of observations and related variables for storing state
    vpi_len = length(v(1,:));
    len = tmax;
    n_obs = floor((dt*(vpi_len-windowlen/dt-1))/stride);%n_obs = floor((len-windowlen)/stride+1);
    state_len = n_obs;

    beta_vec = zeros(1,state_len);
    error_index = zeros(1,state_len);
    time = 0:dt:len;
    timespike = find_SMC_spikes(SMC,dt);
    j = 0; 
    index = 1;

    % loop for calculating state variables
    while index <= length(beta_vec)
        % calculate indexes needed to calculate state
        beginning = vpi_len - (n_obs-index)*stride/dt - windowlen/dt; 
        ending = vpi_len - (n_obs-index)*stride/dt; 

        % window GPi data and TH data
        vpi_c = v(:,beginning:ending); 
        vth_c = vth(:,beginning:ending);

        % Calculate beta from GPi data
        [GPi_APs] = find_spike_times(vpi_c,time,n);
        [GPi_area, ~, ~]=make_Spectrum(GPi_APs,params);
        beta_vec(index) = GPi_area;
        
        % Calculate error index
        t_hold = (beginning*dt):dt:(ending*dt); % t_hold = ((vpi_len-len/dt-(len-j)*100)*dt):dt:((vpi_len-(len-j)*100)*dt);
        ts_c = timespike(timespike>t_hold(1));
        ts_c = ts_c(ts_c<t_hold(end));
        error_index(index)=calculateEI(t_hold,vth_c,ts_c,t_hold(1),t_hold(end));
        j = j + stride;
        index = index + 1;
    end
end

function spike = find_SMC_spikes(SMC,dt)
% finds the spike times in ms of the SMC
spike = [];
if SMC(1)>0
    spike = [spike 1*dt];
end
for i=2:length(SMC)
    if SMC(i)>0 && abs(SMC(i-1))<dt
        spike = [spike i*dt];
    end
end
end

function er=calculateEI(t,vth,timespike,tstart,tend)
%Calculates the Error Index (EI)

%Input:
%t - time vector (msec)
%vth - Array with membrane potentials of each thalamic cell
%timespike - Time of each SMC input pulse
%tmax - maximum time taken into consideration for calculation

%Output:
%er - Error index

m=size(vth,1);
tmax = tend;

e=zeros(1,m);
b1=find(timespike>=tstart, 1 ); % stop ignoring first 200 seconds 
b2=find(timespike<=tend-25, 1, 'last' ); %ignore last 25 msec

for i=1:m
    clear compare a b
    compare=[];
    k=1;
    for j=2:length(vth(i,:)) 
        if vth(i,j-1)<-40 && vth(i,j)>-40
            compare(k)=t(j);
            k=k+1;
        end
    end
    for p=b1:b2
        if p~=b2
            a=find(compare>=timespike(p) & compare<timespike(p)+25);
            b=find(compare>=timespike(p)+25 & compare<timespike(p+1));
        elseif b2==length(timespike)
            a=find(compare>=timespike(p) & compare<tmax);
            b=[];
        else
            a=find(compare>=timespike(p) & compare<timespike(p+1));
            b=find(compare>=timespike(p)+25 & compare<timespike(p+1));
        end
        if isempty(a)
            e(i)=e(i)+1;
        elseif size(a,2)>1
            e(i)=e(i)+1;
        end
        if ~isempty(b)
            e(i)=e(i)+length(b);
        end
    end

end
er=mean(e/(b2-b1+1));
end

function [beta,w,SS] = PSD_calc(data,params,n)
    fs = params.Fs;
    if mod(length(data),2)
        % we want to make this even. taking off first point.
        data = data(:,2:end);
    end
    fftdata = zeros(size(data));
    int = zeros([1,n]);
    S = zeros([n,(length(fftdata)+2)/2]);
    for i = 1:n
        fftdata = fft(data(i,:)); % take fft
        fftdata = fftdata(1:floor(length(fftdata)./2)+1); % 1st half
        psddata = (1/(fs.*length(fftdata)))*abs(fftdata).^2;
        psddata(2:end-1) = 2.*psddata(2:end-1);
        S(i,:) = psddata;
        w = linspace(0,fs/2,length(psddata));
        windowfft = psddata(w>13 & w<35);
        int(i) = trapz(w(w>13 & w<35),windowfft);
    end
    SS = mean(S,1);
    SS = mean(SS(w>13 & w<35));
    beta = mean(int);
end

function [EI] = EI_calc(voltage,pulse,n,dt)
    % find fs
    lookup_length = 40; % ms
    TH_pulse_height = -40; % mV

    % setup 
    end_length = -1;
    SMC_pulse = 0;
    TH_miss = 0;

    % loop through each neuron to calculate
    for i = 1:length(voltage)
        if pulse(i) < 0.1 || i<end_length
            continue
        end
        % we have an SMC pulse
        SMC_pulse = SMC_pulse + 1;
        end_length = i + (lookup_length/dt);
        if end_length>length(voltage)
            end_length = length(voltage);
        end

        % check for misses in the th
            for j = 1:n
                if max(voltage(j,i:end_length))<TH_pulse_height
                    % miss
                    TH_miss = TH_miss+1;
                end
            end
    end

    EI = TH_miss./(n.*SMC_pulse);
end

% function [beta,w,SS] = PSD_calc(data,params,n)
%     fs = params.Fs;
%     if mod(length(data),2)
%         % we want to make this even. taking off first point.
%         data = data(:,2:end);
%     end
%     fftdata = zeros(size(data));
%     int = zeros([1,n]);
%     S = zeros([n,(length(fftdata)+2)/2]);
%     for i = 1:n
%         fftdata = fft(data(i,:)); % take fft
%         fftdata = fftdata(1:floor(length(fftdata)./2)+1); % 1st half
%         psddata = (1/(fs.*length(fftdata)))*abs(fftdata).^2;
%         psddata(2:end-1) = 2.*psddata(2:end-1);
%         S(i,:) = psddata;
%         w = linspace(0,fs/2,length(psddata));
%         windowfft = psddata(w>13 & w<35);
%         int(i) = trapz(windowfft);
%     end
%     SS = mean(S,1);
%     SS = mean(SS(w>13 & w<35));
%     beta = mean(int);
% end

function [data] = find_spike_times(v,t,nn)


    data(1:nn) = struct('times',[]);
    t = t./1000;    % unit: second
    for k = 1:nn
        data(k).times = t(diff(v(k,:)>-20)==1)';
    end

end

function [ainf] = gpe_ainf(V)
    ainf=1./(1+exp(-(V+57)./2));
end

function [hinf] = gpe_hinf(V)
    hinf=1./(1+exp((V+58)./12));
end

function [minf] = gpe_minf(V)
    minf=1./(1+exp(-(V+37)./10));
end

function [ninf] = gpe_ninf(V)
    ninf=1./(1+exp(-(V+50)./14));
end

function [rinf] = gpe_rinf(V)
    rinf=1./(1+exp((V+70)./2));
end

function [sinf] = gpe_sinf(V)
    sinf=1./(1+exp(-(V+35)./2));
end

function [tau] = gpe_tauh(V)
    tau=0.05+0.27./(1+exp(-(V+40)./-12));
end

function [tau] = gpe_taun(V)
    tau=0.05+0.27./(1+exp(-(V+40)./-12));
end

function [hinf] = th_hinf(V)
    hinf=1./(1+exp((V+41)./4));
end

function [minf] = th_minf(V)
    minf=1./(1+exp(-(V+37)./7));
end

function [pinf] = th_pinf(V)
    pinf=1./(1+exp(-(V+60)./6.2));
end

function [rinf] = th_rinf(V)
    rinf=1./(1+exp((V+84)./4));
end

function [tau] = th_tauh(V)
    tau=1./(ah(V)+bh(V));
end

function [a] = ah(V)
    a=0.128*exp(-(V+46)./18); % part of th_tauh fxn
end

function [b] = bh(V)
    b=4./(1+exp(-(V+23)./5)); % part of th_tauh fxn
end

function [tau] = th_taur(V)
    tau=0.15*(28+exp(-(V+25)./10.5));
end

function [ah] = alphah(V)
ah=0.128*exp((-50-V)/18);
end

function [am] = alpham(V)
am=(0.32*(54+V))./(1-exp((-54-V)/4));
end

function [an] = alphan(V)
an=(0.032*(52+V))./(1-exp((-52-V)./5));
end

function [ap] = alphap(V)
ap=(3.209*10^-4*(30+V))./(1-exp((-30-V)./9));
end

function [bh] = betah(V)
bh=4./(1+exp((-27-V)/5));
end

function [bn] = betan(V)
bn=0.5*exp((-57-V)./40);
end

function [bm] = betam(V)
bm=0.28*(V+27)./((exp((27+V)/5))-1);
end

function [bp] = betap(V)
bp=(-3.209*10^-4*(30+V))./(1-exp((30+V)./9));
end

function [gb] = Ggaba(V)
gb=2*(1+tanh(V/4));
end

function [ainf] = stn_ainf(V)
    ainf=1./(1+exp(-(V+45)./14.7));
end

function [binf] = stn_binf(V)
    binf=1./(1+exp((V+90)./7.5));
end

function [cinf] = stn_cinf(V)
    cinf=1./(1+exp(-(V+30.6)./5));
end

function [d1inf] = stn_d1inf(V)
    d1inf=1./(1+exp((V+60)./7.5));
end

function [d2inf] = stn_d2inf(V)
    d2inf=1./(1+exp((V-0.1)./0.02));
end

function [hinf] = stn_hinf(V)
    hinf=1./(1+exp((V+45.5)./6.4));
end

function [minf] = stn_minf(V)
    minf=1./(1+exp(-(V+40)./8));
end

function [ninf] = stn_ninf(V)
    ninf=1./(1+exp(-(V+41)./14));
end

function [pinf] = stn_pinf(V)
    pinf=1./(1+exp(-(V+56)./6.7));
end

function [qinf] = stn_qinf(V)
    qinf=1./(1+exp((V+85)./5.8));
end

function [rinf] = stn_rinf(V)
    rinf=1./(1+exp(-(V-0.17)./0.08));
end

function [tau] = stn_taua(V)
    tau=1+1./(1+exp(-(V+40)./-0.5));
end

function [tau] = stn_taub(V)
    tau=200./(exp(-(V+60)./-30)+exp(-(V+40)./10));
end

function [tau] = stn_tauc(V)
    tau=45+10./(exp(-(V+27)./-20)+exp(-(V+50)./15));
end

function [tau] = stn_taud1(V)
    tau=400+500./(exp(-(V+40)./-15)+exp(-(V+20)./20));
end

function [tau] = stn_tauh(V)
    tau=24.5./(exp(-(V+50)./-15)+exp(-(V+50)./16));
end

function [tau] = stn_taum(V)
    tau=0.2+3./(1+exp(-(V+53)./-0.7));
end

function [tau] = stn_taun(V)
    tau=11./(exp(-(V+40)./-40)+exp(-(V+40)./50));
end

function [tau] = stn_taup(V)
    tau=5+0.33./(exp(-(V+27)./-10)+exp(-(V+102)./15));
end

function [tau] = stn_tauq(V)
    tau=400./(exp(-(V+50)./-15)+exp(-(V+50)./16));
end

function [area,S,f] = make_Spectrum(raw,params)

% Compute Multitaper Spectrum
[S,f] = mtspectrumpt(raw,params);
beta = S(f>13 & f<35);
betaf = f(f>13 & f<35);
area = trapz(betaf,beta);

end

function [S,f,R,Serr]=mtspectrumpt(data,params,fscorr,t)

if nargin < 1; error('Need data'); end
if nargin < 2; params=[]; end
[tapers,pad,Fs,fpass,err,trialave,params]=getparams(params);
clear params
data=change_row_to_column(data);
if nargout > 3 && err(1)==0; error('cannot compute error bars with err(1)=0; change params and run again'); end
if nargin < 3 || isempty(fscorr); fscorr=0;end
if nargin < 4 || isempty(t)
   [mintime,maxtime]=minmaxsptimes(data);
   dt=1/Fs; % sampling time
   t=mintime-dt:dt:maxtime+dt; % time grid for prolates
end
N=length(t); % number of points in grid for dpss
nfft=max(2^(nextpow2(N)+pad),N); % number of points in fft of prolates
[f,findx]=getfgrid(Fs,nfft,fpass); % get frequency grid for evaluation
tapers=dpsschk(tapers,N,Fs); % check tapers
[J,Msp,Nsp]=mtfftpt(data,tapers,nfft,t,f,findx); % mt fft for point process times
S=squeeze(mean(conj(J).*J,2));
if trialave; S=squeeze(mean(S,2));Msp=mean(Msp);end
R=Msp*Fs;
if nargout==4
   if fscorr==1
      Serr=specerr(S,J,err,trialave,Nsp);
   else
      Serr=specerr(S,J,err,trialave);
   end
end
end

function [tapers,pad,Fs,fpass,err,trialave,params]=getparams(params)

if ~isfield(params,'tapers') || isempty(params.tapers)  %If the tapers don't exist
     display('tapers unspecified, defaulting to params.tapers=[3 5]');
     params.tapers=[3 5];
end
if ~isempty(params) && length(params.tapers)==3 
    % Compute timebandwidth product
    TW = params.tapers(2)*params.tapers(1);
    % Compute number of tapers
    K  = floor(2*TW - params.tapers(3));
    params.tapers = [TW  K];
end

if ~isfield(params,'pad') || isempty(params.pad)
    params.pad=0;
end
if ~isfield(params,'Fs') || isempty(params.Fs)
    params.Fs=1;
end
if ~isfield(params,'fpass') || isempty(params.fpass)
    params.fpass=[0 params.Fs/2];
end
if ~isfield(params,'err') || isempty(params.err)
    params.err=0;
end
if ~isfield(params,'trialave') || isempty(params.trialave)
    params.trialave=0;
end

tapers=params.tapers;
pad=params.pad;
Fs=params.Fs;
fpass=params.fpass;
err=params.err;
trialave=params.trialave;
end

function data=change_row_to_column(data)

dtmp=[];
if isstruct(data)
   C=length(data);
   if C==1
      fnames=fieldnames(data);
      eval(['dtmp=data.' fnames{1} ';'])
      data=dtmp(:);
   end
else
  [N,C]=size(data);
  if N==1 || C==1
    data=data(:);
  end
end
end

function [mintime, maxtime]=minmaxsptimes(data)

dtmp='';
if isstruct(data)
   data=reshape(data,numel(data),1);
   C=size(data,1);
   fnames=fieldnames(data);
   mintime=zeros(1,C); maxtime=zeros(1,C);
   for ch=1:C
     eval(['dtmp=data(ch).' fnames{1} ';'])
     if ~isempty(dtmp)
        maxtime(ch)=max(dtmp);
        mintime(ch)=min(dtmp);
     else
        mintime(ch)=NaN;
        maxtime(ch)=NaN;
     end
   end
   maxtime=max(maxtime); % maximum time
   mintime=min(mintime); % minimum time
else
     dtmp=data;
     if ~isempty(dtmp)
        maxtime=max(dtmp);
        mintime=min(dtmp);
     else
        mintime=NaN;
        maxtime=NaN;
     end
end
if mintime < 0 
   error('Minimum spike time is negative'); 
end
end

function [f,findx]=getfgrid(Fs,nfft,fpass)

if nargin < 3; error('Need all arguments'); end
df=Fs/nfft;
f=0:df:Fs; % all possible frequencies
f=f(1:nfft);
if length(fpass)~=1
   findx=find(f>=fpass(1) & f<=fpass(end));
else
   [fmin,findx]=min(abs(f-fpass));
   clear fmin
end
f=f(findx);
end

function [tapers,eigs]=dpsschk(tapers,N,Fs)

if nargin < 3; error('Need all arguments'); end
sz=size(tapers);
if sz(1)==1 && sz(2)==2
    [tapers,eigs]=dpss(N,tapers(1),tapers(2));
    tapers = tapers*sqrt(Fs);
elseif N~=sz(1)
    error('seems to be an error in your dpss calculation; the number of time points is different from the length of the tapers');
end
end

function [J,Msp,Nsp]=mtfftpt(data,tapers,nfft,t,f,findx)

if nargin < 6; error('Need all input arguments'); end
if isstruct(data); C=length(data); else C=1; end% number of channels
K=size(tapers,2); % number of tapers
nfreq=length(f); % number of frequencies
if nfreq~=length(findx); error('frequency information (last two arguments) inconsistent'); end
H=fft(tapers,nfft,1);  % fft of tapers
H=H(findx,:); % restrict fft of tapers to required frequencies
w=2*pi*f; % angular frequencies at which ft is to be evaluated
Nsp=zeros(1,C); Msp=zeros(1,C);
for ch=1:C
  if isstruct(data)
     fnames=fieldnames(data);
     eval(['dtmp=data(ch).' fnames{1} ';'])
     indx=find(dtmp>=min(t)&dtmp<=max(t));
     if ~isempty(indx); dtmp=dtmp(indx);
     end
  else
     dtmp=data;
     indx=find(dtmp>=min(t)&dtmp<=max(t));
     if ~isempty(indx); dtmp=dtmp(indx);
     end
  end
  Nsp(ch)=length(dtmp);
  Msp(ch)=Nsp(ch)/length(t);
  if Msp(ch)~=0
      data_proj=interp1(t',tapers,dtmp);
      exponential=exp(-1i*w'*(dtmp-t(1))');
      J(:,:,ch)=exponential*data_proj-H*Msp(ch);
  else
      J(1:nfreq,1:K,ch)=0;
  end
end
end

function Serr=specerr(S,J,err,trialave,numsp)
 
if nargin < 4; error('Need at least 4 input arguments'); end
if err(1)==0; error('Need err=[1 p] or [2 p] for error bar calculation. Make sure you are not asking for the output of Serr'); end
[nf,K,C]=size(J);
errchk=err(1);
p=err(2);
pp=1-p/2;
qq=1-pp;

if trialave
   dim=K*C;
   C=1;
   dof=2*dim;
   if nargin==5; dof = fix(1/(1/dof + 1/(2*sum(numsp)))); end
   J=reshape(J,nf,dim);
else
   dim=K;
   dof=2*dim*ones(1,C);
   for ch=1:C
     if nargin==5; dof(ch) = fix(1/(1/dof + 1/(2*numsp(ch)))); end 
   end
end
Serr=zeros(2,nf,C);
if errchk==1
   Qp=chi2inv(pp,dof);
   Qq=chi2inv(qq,dof);
   Serr(1,:,:)=dof(ones(nf,1),:).*S./Qp(ones(nf,1),:);
   Serr(2,:,:)=dof(ones(nf,1),:).*S./Qq(ones(nf,1),:);
elseif errchk==2
   tcrit=tinv(pp,dim-1);
   for k=1:dim
       indices=setdiff(1:dim,k);
       Jjk=J(:,indices,:); % 1-drop projection
       eJjk=squeeze(sum(Jjk.*conj(Jjk),2));
       Sjk(k,:,:)=eJjk/(dim-1); % 1-drop spectrum
   end
   sigma=sqrt(dim-1)*squeeze(std(log(Sjk),1,1)); if C==1; sigma=sigma'; end 
   conf=repmat(tcrit,nf,C).*sigma;
   conf=squeeze(conf); 
   Serr(1,:,:)=S.*exp(-conf); Serr(2,:,:)=S.*exp(conf);
end
Serr=squeeze(Serr);
end

