import { useState, useEffect, useCallback, useMemo } from "react";

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════════════════════════════ */
const MAKES = ["Acura","Alfa Romeo","Audi","BMW","Buick","Cadillac","Chevrolet","Chrysler","Dodge","Fiat","Ford","GMC","Honda","Hummer","Hyundai","Infiniti","Jaguar","Jeep","Kia","Land Rover","Lexus","Lincoln","Maserati","Mazda","Mercedes-Benz","Mini","Mitsubishi","Nissan","Pontiac","Porsche","Ram","Subaru","Tesla","Toyota","Volkswagen","Volvo"];
const BODY_TYPES = ["Convertible","Coupe","Hatchback","Minivan","Pickup","Sedan","SUV","Wagon"];
const DRIVE_TYPES = ["AWD","FWD","RWD","4WD"];
const YEARS = Array.from({length:47},(_,i)=>2026-i);
const TIER={"Mercedes-Benz":"Luxury",BMW:"Luxury",Audi:"Luxury",Lexus:"Luxury",Porsche:"Luxury",Jaguar:"Luxury","Land Rover":"Luxury",Cadillac:"Luxury",Volvo:"Luxury",Acura:"Luxury",Infiniti:"Luxury",Lincoln:"Luxury",Maserati:"Luxury",Tesla:"Luxury","Alfa Romeo":"Luxury",Hummer:"Luxury",Toyota:"Mid",Honda:"Mid",Volkswagen:"Mid",Subaru:"Mid",Mazda:"Mid",Hyundai:"Mid",Kia:"Mid",Buick:"Mid",Mini:"Mid",GMC:"Mid",Chrysler:"Mid",Nissan:"Mid",Ram:"Mid",Jeep:"Mid",Dodge:"Mid",Ford:"Mid",Chevrolet:"Mid",Fiat:"Economy",Mitsubishi:"Economy",Pontiac:"Economy"};
const MODELS={Toyota:["RAV4","Camry","Corolla","Highlander","4Runner","Tacoma","Prius"],Honda:["CR-V","Civic","Accord","Pilot","HR-V","Odyssey"],Ford:["F-150","Escape","Explorer","Mustang","Fusion","Bronco"],Chevrolet:["Silverado","Equinox","Malibu","Traverse","Tahoe","Camaro"],BMW:["3 Series","X3","X5","5 Series","X1"],Hyundai:["Tucson","Elantra","Santa Fe","Sonata","Kona"],Subaru:["Outback","Forester","Crosstrek","Impreza","WRX"],Nissan:["Rogue","Altima","Sentra","Pathfinder","Frontier"],Jeep:["Wrangler","Grand Cherokee","Cherokee","Compass"],Mazda:["CX-5","Mazda3","CX-9","MX-5 Miata"],Kia:["Sportage","Forte","Seltos","Telluride","K5"],Lexus:["RX","ES","NX","IS"],Audi:["A4","Q5","A3","Q7"],"Mercedes-Benz":["C-Class","GLC","E-Class","GLE"],Volkswagen:["Jetta","Tiguan","Atlas","Golf"],Acura:["MDX","TLX","RDX"],Volvo:["XC60","XC90","S60"],Dodge:["Charger","Challenger","Durango"],Ram:["1500","2500"],GMC:["Sierra","Terrain","Acadia","Yukon"],Buick:["Encore","Enclave"],Cadillac:["XT5","Escalade","CT5"],Lincoln:["Corsair","Aviator"],Porsche:["Cayenne","Macan","911"],Tesla:["Model 3","Model Y","Model S"],Infiniti:["QX60","Q50"],Chrysler:["Pacifica","300"],"Land Rover":["Range Rover Sport","Discovery"],Jaguar:["F-PACE","XF"],Mini:["Cooper","Countryman"],Maserati:["Ghibli","Levante"],"Alfa Romeo":["Giulia","Stelvio"],Mitsubishi:["Outlander","Eclipse Cross"],Fiat:["500","500X"],Pontiac:["G6","GTO"]};
const BODY_MAP={SUV:["RAV4","CR-V","Escape","Equinox","X3","X5","Tucson","Santa Fe","Outback","Forester","Crosstrek","Rogue","Pathfinder","Wrangler","Grand Cherokee","Cherokee","Compass","CX-5","CX-9","Sportage","Seltos","Telluride","RX","NX","Q5","Q7","GLC","GLE","Tiguan","Atlas","MDX","RDX","XC60","XC90","Durango","Terrain","Acadia","Yukon","Encore","Enclave","XT5","Corsair","Aviator","Cayenne","Macan","Model Y","QX60","Discovery","F-PACE","Countryman","Levante","Stelvio","Outlander","Eclipse Cross","500X","Highlander","4Runner","Pilot","HR-V","Explorer","Bronco","Traverse","Tahoe","Kona","Escalade"],Sedan:["Camry","Corolla","Civic","Accord","Malibu","3 Series","5 Series","Elantra","Sonata","Impreza","Altima","Sentra","Mazda3","Forte","K5","ES","IS","A4","A3","C-Class","E-Class","Jetta","Golf","TLX","S60","Charger","CT5","Q50","300","XF","Ghibli","Giulia","G6","Prius","Fusion","Model 3","Model S","WRX"],Pickup:["F-150","Silverado","Tacoma","Frontier","1500","2500","Sierra"],Coupe:["Mustang","MX-5 Miata","Camaro","Challenger","911","GTO","500"],Minivan:["Odyssey","Pacifica"],Hatchback:["Golf","Impreza","Mazda3","Crosstrek","Prius","Cooper","Corolla"],Wagon:["Outback"],Convertible:["MX-5 Miata","Mustang","911","500"]};
const REGIONS=["Northeast","Southeast","Midwest","Southwest","West Coast","Mountain","Mid-Atlantic","Pacific NW"];
// Depreciation rates by tier (annual % loss)
const DEPR_RATE={Luxury:0.095,Mid:0.072,Economy:0.065};
// Regional price multipliers
const REGION_MULT={Northeast:1.04,Southeast:0.96,Midwest:0.94,Southwest:0.98,"West Coast":1.08,Mountain:0.97,"Mid-Atlantic":1.03,"Pacific NW":1.05};

/* ═══════════════════════════════════════════════════════════════
   SIMULATION: Direction 1 - Price Prediction
   ═══════════════════════════════════════════════════════════════ */


async function fetchSellerPrice(data){
  const res = await fetch("http://127.0.0.1:8000/api/seller/price", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      make: data.make,
      model: data.model,
      year: Number(data.year),
      mileage: Number(data.mileage),
      body_type: data.body,
      drive_type: data.drive,
      zipcode: data.zip || "",
      engine: data.engine || "",
      trim: data.trim || "",
    }),
  });

  if (!res.ok) {
    throw new Error("Seller API request failed");
  }

  const dataJson = await res.json();
  return {
    low: Math.round(dataJson.competitive_price),
    mid: Math.round(dataJson.fair_market_price),
    high: Math.round(dataJson.premium_price),
    raw: dataJson,
  };
}
function simPrice(make,model,year,mileage,bodyType,driveType,zip){
  const tier=TIER[make]||"Economy";
  const b0={Luxury:40000,Mid:23000,Economy:14000}[tier];
  const bm={SUV:1.22,Pickup:1.2,Sedan:1.0,Coupe:1.08,Convertible:1.12,Hatchback:0.93,Minivan:0.96,Wagon:0.98};
  const dm={AWD:1.07,"4WD":1.05,FWD:1.0,RWD:1.02};
  const age=Math.max(2026-year,0);
  let p=b0*(bm[bodyType]||1)*(dm[driveType]||1);
  p*=Math.max(1-age*0.062,0.12);
  p*=Math.max(1-mileage*0.0000024,0.28);
  const mh=(model||"").split("").reduce((a,c)=>a+c.charCodeAt(0),0);
  p*=0.92+(mh%20)/100;
  // regional adjustment
  const zipNum=parseInt((zip||"500").substring(0,3))||500;
  const regAdj=zipNum<200?1.04:zipNum<400?1.02:zipNum<600?0.96:zipNum<800?0.98:1.06;
  p*=regAdj;
  const mid=Math.round(Math.max(p,1200));
  return{low:Math.round(mid*0.83),mid,high:Math.round(mid*1.17)};
}

/* ═══════════════════════════════════════════════════════════════
   SIMULATION: Direction 3 - Depreciation & Regional
   ═══════════════════════════════════════════════════════════════ */
function simDepreciation(make,model,bodyType,currentAge){
  const tier=TIER[make]||"Economy";
  const rate=DEPR_RATE[tier]||0.072;
  // Generate depreciation curve points (age 0-15)
  const curve=[];
  const newPrice=simPrice(make,model,2026,0,bodyType,"AWD","500").mid/Math.max(1-0*0.062,0.12)*1;
  const baseNew=newPrice*1.5;
  for(let a=0;a<=15;a++){
    const depFactor=Math.pow(1-rate,a)*(a===0?1:Math.max(1-a*0.015,0.7));
    const extraDrop=a<=3?a*0.04:0.12;
    curve.push({age:a,price:Math.round(baseNew*(depFactor-extraDrop)),pctRemaining:Math.round((depFactor-extraDrop)*100)});
  }
  // Determine stage
  let stage,stageColor,stageDesc;
  if(currentAge<=2){stage="Steep Drop";stageColor="#ef5555";stageDesc="Your car is in the steepest depreciation phase. New cars typically lose 15-25% in the first 2 years.";}
  else if(currentAge<=5){stage="Moderate Decline";stageColor="#e8a83e";stageDesc="Depreciation is slowing but still significant. This is often a good time to sell before the curve flattens.";}
  else if(currentAge<=10){stage="Gradual Plateau";stageColor="#4e9af5";stageDesc="Depreciation has slowed considerably. The car retains value more steadily at this stage.";}
  else{stage="Stable Floor";stageColor="#34c77b";stageDesc="The car has lost most of its depreciation. Remaining value is relatively stable.";}
  // Future projection
  const currentVal=curve.find(c=>c.age===Math.min(currentAge,15))?.price||curve[curve.length-1].price;
  const futureAge=Math.min(currentAge+2,15);
  const futureVal=curve.find(c=>c.age===futureAge)?.price||curve[curve.length-1].price;
  const lossIn2Years=currentVal-futureVal;
  const lossPct=currentVal>0?Math.round(lossIn2Years/currentVal*100):0;
  // Brand comparison
  const holdValueRank=tier==="Luxury"?"Below Average":tier==="Mid"?"Good":"Average";
  return{curve,stage,stageColor,stageDesc,currentVal,futureVal,lossIn2Years,lossPct,holdValueRank,rate:Math.round(rate*100)};
}

function simRegionalPrices(make,model,year,mileage,bodyType,driveType){
  const results=REGIONS.map(region=>{
    const mult=REGION_MULT[region]||1.0;
    const noise=0.97+Math.random()*0.06;
    const basePrice=simPrice(make,model,year,mileage,bodyType,driveType,"500").mid;
    const regionPrice=Math.round(basePrice*mult*noise);
    return{region,price:regionPrice,diff:Math.round((mult-1)*100),isCheaper:mult<1};
  });
  results.sort((a,b)=>a.price-b.price);
  return results;
}

/* ═══════════════════════════════════════════════════════════════
   SIMULATION: Direction 2 - Buyer Recommendations
   ═══════════════════════════════════════════════════════════════ */

async function fetchBuyerRecommendations(query){
  const res = await fetch("http://127.0.0.1:8000/api/buyer/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      budget: Number(query.budget),
      body_type: query.body_type || null,
      drive_type: query.drive_type || null,
      make: query.make || null,
      max_mileage: query.max_mileage ? Number(query.max_mileage) : null,
      min_year: query.min_year ? Number(query.min_year) : null,
      zipcode: query.zipcode || null,
      top_n: Number(query.top_n || 10),
    }),
  });

  if (!res.ok) {
    throw new Error("Buyer API request failed");
  }

  const dataJson = await res.json();
  const rows = Array.isArray(dataJson.results) ? dataJson.results : [];

  return rows.map((row, index) => {
    const make = row.make || row.brand_clean || row.Make || "";
    const model = row.model || row.Model || row.title || "";
    const title = row.title || `${make} ${model}`.trim() || `Option ${index + 1}`;

    const yearMin = row.year_min ?? row.min_year ?? row.Year_min ?? row.yearLow;
    const yearMax = row.year_max ?? row.max_year ?? row.Year_max ?? row.yearHigh;
    const typicalYear = Number(row.typical_year ?? row.year ?? row.median_year ?? row.Year ?? yearMax ?? yearMin ?? 0);
    const yearRange = row.year_range || (yearMin && yearMax ? `${yearMin}–${yearMax}` : typicalYear ? String(typicalYear) : "—");

    const typicalMileage = Math.round(Number(row.typical_mileage ?? row.mileage ?? row.median_mileage ?? row.Mileage ?? 0));
    const typicalPrice = Math.round(Number(row.typical_price ?? row.price ?? row.actual_price ?? row.avg_price ?? row.pricesold ?? 0));
    const predictedFair = Math.round(Number(row.predicted_fair ?? row.fair_market_price ?? row.mid ?? row.predicted_mid ?? 0));
    const predictedCompetitive = Math.round(Number(row.predicted_competitive ?? row.competitive_price ?? row.low ?? row.predicted_low ?? (predictedFair ? predictedFair * 0.83 : 0)));
    const predictedPremium = Math.round(Number(row.predicted_premium ?? row.premium_price ?? row.high ?? row.predicted_high ?? (predictedFair ? predictedFair * 1.17 : 0)));

    const rawValuePct = row.avg_value_pct ?? row.value_pct ?? row.value_percent ?? row.avg_value_percent;
    const avgValuePct = rawValuePct !== undefined && rawValuePct !== null
      ? Number(rawValuePct)
      : (predictedFair > 0 ? ((predictedFair - typicalPrice) / predictedFair) * 100 : 0);

    const sampleCount = Number(row.sample_count ?? row.samples ?? row.count ?? row.n ?? 0);
    const confidence = row.confidence || (sampleCount >= 100 ? "High" : sampleCount >= 30 ? "Medium" : "Low");

    const bodyType = row.body_type || row.BodyType || query.body_type || "—";
    const driveType = row.drive_type || row.DriveType || query.drive_type || "—";

    const tier = TIER[make] || "Economy";
    const dRate = DEPR_RATE[tier] || 0.072;
    const deprLabel = dRate < 0.07 ? "Holds value well" : dRate < 0.085 ? "Average depreciation" : "Depreciates faster";
    const deprColor = dRate < 0.07 ? "#34c77b" : dRate < 0.085 ? "#e8a83e" : "#ef5555";

    const dealLabel = avgValuePct >= 12 ? "Great Deal" : avgValuePct >= 5 ? "Good Deal" : avgValuePct >= 0 ? "Fair Deal" : "Above Market";
    const dealColor = avgValuePct >= 12 ? "#34c77b" : avgValuePct >= 5 ? "#4e9af5" : avgValuePct >= 0 ? "#e8a83e" : "#ef5555";

    const reason = row.reason || (
      avgValuePct >= 10
        ? "Strong value relative to the model’s fair-value estimate."
        : avgValuePct >= 4
        ? "Priced a bit below fair value with solid overall fundamentals."
        : "Reasonable match for the selected budget and filters."
    );

    return {
      ...row,
      make,
      model,
      title,
      typical_year: typicalYear,
      year_range: yearRange,
      typical_mileage: typicalMileage,
      body_type: bodyType,
      drive_type: driveType,
      typical_price: typicalPrice,
      predicted_fair: predictedFair,
      predicted_competitive: predictedCompetitive,
      predicted_premium: predictedPremium,
      avg_value_pct: avgValuePct,
      sample_count: sampleCount,
      confidence,
      reason,
      dealLabel,
      dealColor,
      deprLabel,
      deprColor,
    };
  });
}

function simRecommend(budget,bodyType,driveType,make,maxMileage,minYear){
  const pool=[];
  const mkList=make?[make]:Object.keys(MODELS);
  mkList.forEach(mk=>{
    (MODELS[mk]||[]).forEach(mdl=>{
      if(bodyType){const m=BODY_MAP[bodyType]||[];if(!m.includes(mdl))return;}
      for(let t=0;t<3;t++){
        const yr=2014+Math.floor(Math.random()*10);
        const ml=10000+Math.floor(Math.random()*95000);
        if(minYear&&yr<minYear)continue;
        if(maxMileage&&ml>maxMileage)continue;
        const bt=bodyType||Object.entries(BODY_MAP).find(([,v])=>v.includes(mdl))?.[0]||"Sedan";
        const dt=driveType||["AWD","FWD","RWD","4WD"][Math.floor(Math.random()*4)];
        const{mid}=simPrice(mk,mdl,yr,ml,bt,dt,"500");
        const tp=Math.round(mid*(0.82+Math.random()*0.22));
        if(tp>budget)continue;
        const vp=((mid-tp)/mid*100);
        if(vp<-5||vp>50)continue;
        const sc=15+Math.floor(Math.random()*200);
        const conf=sc>=100?"High":sc>=30?"Medium":"Low";
        // Direction 3: depreciation label
        const tier=TIER[mk]||"Economy";
        const dRate=DEPR_RATE[tier]||0.072;
        const deprLabel=dRate<0.07?"Holds value well":dRate<0.085?"Average depreciation":"Depreciates faster";
        const deprColor=dRate<0.07?"#34c77b":dRate<0.085?"#e8a83e":"#ef5555";
        // Deal quality
        const dealLabel=vp>=12?"Great Deal":vp>=5?"Good Deal":vp>=0?"Fair Deal":"Above Market";
        const dealColor=vp>=12?"#34c77b":vp>=5?"#4e9af5":vp>=0?"#e8a83e":"#ef5555";
        const reasons=[];
        if(vp>=10)reasons.push("priced well below fair-value estimate");
        else if(vp>=4)reasons.push("slightly below fair-value estimate");
        if(ml<60000)reasons.push("relatively low mileage");
        if(yr>=2019)reasons.push("recent model year");
        if(dRate<0.07)reasons.push("strong resale value");
        if(conf==="High")reasons.push("backed by many comparable sales");
        if(!reasons.length)reasons.push("balanced match for your preferences");
        pool.push({make:mk,model:mdl,title:`${mk} ${mdl}`,body_type:bt,drive_type:dt,
          year_range:`${yr-1}\u2013${yr+2}`,typical_year:yr,typical_mileage:ml,
          typical_price:tp,predicted_fair:mid,
          avg_value_pct:Math.round(vp*10)/10,sample_count:sc,confidence:conf,
          buyer_score:(vp*0.45+(1-ml/120000)*20+sc/200*20)/100,
          reason:reasons.join("; "),dealLabel,dealColor,deprLabel,deprColor,
          price_low:Math.round(mid*0.83),price_high:Math.round(mid*1.17)});
      }
    });
  });
  const best={};
  pool.forEach(r=>{if(!best[r.title]||r.buyer_score>best[r.title].buyer_score)best[r.title]=r;});
  return Object.values(best).sort((a,b)=>b.buyer_score-a.buyer_score).slice(0,10);
}

/* ═══════════════════════════════════════════════════════════════
   GLOBAL STYLES
   ═══════════════════════════════════════════════════════════════ */
const CSS=`
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root{--bg0:#0b0c0f;--bg1:#10121a;--bg2:#161923;--bg3:#1c2030;--bg4:#242939;--bd1:#232840;--bd2:#2d3350;--t0:#eef0f6;--t1:#b8bdd0;--t2:#7d84a0;--t3:#525878;--amber:#dea03a;--amberDim:#dea03a20;--amberMid:#dea03a45;--green:#2fc872;--greenDim:#2fc87218;--red:#e85050;--redDim:#e8505018;--blue:#4a90e8;--blueDim:#4a90e818;}
*{box-sizing:border-box;margin:0;padding:0;}
@keyframes fadeIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
@keyframes scaleIn{from{opacity:0;transform:scale(0.93)}to{opacity:1;transform:scale(1)}}
@keyframes spin{to{transform:rotate(360deg)}}
input:focus,select:focus{border-color:var(--amber)!important;box-shadow:0 0 0 3px var(--amberDim)!important;outline:none;}
select option{background:var(--bg2);color:var(--t0);}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:3px}
`;
const inp={padding:"11px 14px",borderRadius:10,border:"1px solid var(--bd1)",background:"var(--bg1)",color:"var(--t0)",fontSize:14,fontFamily:"'Outfit',sans-serif",fontWeight:500,outline:"none",transition:"all 0.2s",width:"100%"};
const sel={...inp,cursor:"pointer",appearance:"none",WebkitAppearance:"none",backgroundImage:`url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23525878' stroke-width='2.5'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E")`,backgroundRepeat:"no-repeat",backgroundPosition:"right 14px center"};

/* ═══════════════════════════════════════════════════════════════
   SHARED COMPONENTS
   ═══════════════════════════════════════════════════════════════ */
function F({label,children,hint}){return(<div style={{display:"flex",flexDirection:"column",gap:5}}><label style={{fontSize:10,fontWeight:700,color:"var(--t2)",textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'JetBrains Mono',monospace"}}>{label}</label>{children}{hint&&<span style={{fontSize:10,color:"var(--t3)",fontStyle:"italic"}}>{hint}</span>}</div>);}
function Btn({children,onClick,v="p",disabled,full,s}){const b={padding:"13px 24px",borderRadius:11,border:"none",cursor:disabled?"not-allowed":"pointer",fontSize:14,fontWeight:700,fontFamily:"'Outfit',sans-serif",transition:"all 0.25s",display:"inline-flex",alignItems:"center",justifyContent:"center",gap:8,width:full?"100%":undefined,opacity:disabled?0.5:1,...s};if(v==="p")Object.assign(b,{background:disabled?"var(--bg4)":"var(--amber)",color:"#0b0c0f",boxShadow:disabled?"none":"0 4px 18px var(--amberDim)"});else if(v==="g")Object.assign(b,{background:"transparent",color:"var(--t2)"});return<button style={b}onClick={onClick}disabled={disabled}>{children}</button>;}
function Badge({t,c}){return<span style={{display:"inline-flex",alignItems:"center",gap:5,padding:"3px 9px",borderRadius:16,fontSize:9,fontWeight:700,background:`${c}18`,color:c,border:`1px solid ${c}30`,letterSpacing:0.8,textTransform:"uppercase",fontFamily:"'JetBrains Mono',monospace"}}><span style={{width:5,height:5,borderRadius:"50%",background:c}}/>{t}</span>;}
function Steps({items,cur}){return<div style={{display:"flex",alignItems:"center",gap:0,marginBottom:32}}>{items.map((s,i)=><div key={i}style={{display:"flex",alignItems:"center"}}><div style={{display:"flex",alignItems:"center",gap:7}}><div style={{width:26,height:26,borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",fontSize:11,fontWeight:700,fontFamily:"'JetBrains Mono',monospace",background:i<=cur?"var(--amber)":"var(--bg3)",color:i<=cur?"#0b0c0f":"var(--t3)"}}>{i+1}</div><span style={{fontSize:11,fontWeight:600,color:i<=cur?"var(--t0)":"var(--t3)",whiteSpace:"nowrap"}}>{s}</span></div>{i<items.length-1&&<div style={{width:36,height:1,background:i<cur?"var(--amber)":"var(--bd1)",margin:"0 10px"}}/>}</div>)}</div>;}
function Spinner({t}){return<div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",padding:60,gap:14}}><div style={{width:40,height:40,borderRadius:"50%",border:"3px solid var(--bd2)",borderTopColor:"var(--amber)",animation:"spin 0.8s linear infinite"}}/><p style={{color:"var(--t2)",fontSize:13}}>{t}</p></div>;}
function Section({title,icon,children,delay=0}){return<div style={{padding:24,borderRadius:16,background:"var(--bg2)",border:"1px solid var(--bd1)",marginBottom:20,animation:`slideUp 0.5s ease ${delay}s both`}}><h3 style={{fontSize:13,fontWeight:700,color:"var(--t1)",marginBottom:16,fontFamily:"'Outfit',sans-serif",display:"flex",alignItems:"center",gap:8}}>{icon&&<span>{icon}</span>}{title}</h3>{children}</div>;}

/* ═══════════════════════════════════════════════════════════════
   PAGE 1: ROLE SELECT
   ═══════════════════════════════════════════════════════════════ */
function RoleSelect({onSelect}){
  const[hov,setHov]=useState(null);
  const roles=[
    {key:"seller",icon:"\uD83D\uDCB0",title:"I'm a Seller",desc:"Get a data-driven price estimate, see where your car sits on the depreciation curve, and compare regional pricing.",
      feat:["Price range estimate (Direction 1)","Depreciation stage analysis (Direction 3)","Regional price comparison (Direction 3)","What-if mileage/year simulator"],
      grad:"linear-gradient(135deg, #dea03a08 0%, #e8505008 100%)",bc:"var(--amber)"},
    {key:"buyer",icon:"\uD83D\uDD0D",title:"I'm a Buyer",desc:"Find the best value used cars within your budget, ranked by value score with depreciation and regional insights.",
      feat:["Top 10 value-ranked recommendations (Direction 2)","Deal quality scoring (Direction 2)","Depreciation & resale labels (Direction 3)","Side-by-side comparison tool"],
      grad:"linear-gradient(135deg, #4a90e808 0%, #2fc87208 100%)",bc:"var(--green)"}];
  return(
    <div style={{minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",padding:24,background:"var(--bg0)"}}>
      <div style={{position:"fixed",inset:0,pointerEvents:"none",overflow:"hidden"}}>
        <div style={{position:"absolute",top:"-15%",left:"-10%",width:600,height:600,borderRadius:"50%",background:"radial-gradient(circle, #dea03a05 0%, transparent 70%)"}}/>
        <div style={{position:"absolute",bottom:"-20%",right:"-5%",width:700,height:700,borderRadius:"50%",background:"radial-gradient(circle, #4a90e805 0%, transparent 70%)"}}/>
      </div>
      <div style={{position:"relative",zIndex:1,maxWidth:820,width:"100%",animation:"fadeIn 0.7s ease"}}>
        <div style={{textAlign:"center",marginBottom:44}}>
          <div style={{fontSize:40,marginBottom:10}}>{"\uD83D\uDE97"}</div>
          <h1 style={{fontSize:36,fontWeight:900,fontFamily:"'Outfit',sans-serif",color:"var(--t0)",letterSpacing:-1.5,marginBottom:8}}>CarPrice</h1>
          <p style={{fontSize:15,color:"var(--t2)",maxWidth:440,margin:"0 auto",lineHeight:1.6}}>Used-car decision support powered by ML pricing, recommendations, and market insights.</p>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
          {roles.map(r=>(
            <div key={r.key}onMouseEnter={()=>setHov(r.key)}onMouseLeave={()=>setHov(null)}onClick={()=>onSelect(r.key)}
              style={{padding:28,borderRadius:18,cursor:"pointer",transition:"all 0.35s cubic-bezier(.4,0,.2,1)",background:r.grad,border:`1px solid ${hov===r.key?r.bc:"var(--bd1)"}`,transform:hov===r.key?"translateY(-5px)":"none",boxShadow:hov===r.key?`0 18px 45px ${r.bc}12`:"0 2px 8px rgba(0,0,0,0.2)"}}>
              <div style={{fontSize:32,marginBottom:14}}>{r.icon}</div>
              <h2 style={{fontSize:20,fontWeight:800,color:"var(--t0)",marginBottom:8}}>{r.title}</h2>
              <p style={{fontSize:12,color:"var(--t2)",lineHeight:1.6,marginBottom:18}}>{r.desc}</p>
              <div style={{display:"flex",flexDirection:"column",gap:7}}>
                {r.feat.map((f,i)=><div key={i}style={{display:"flex",alignItems:"flex-start",gap:7,fontSize:11,color:"var(--t1)",fontWeight:500}}><span style={{color:r.bc,fontSize:12,flexShrink:0,marginTop:1}}>{"\u2192"}</span><span>{f}</span></div>)}
              </div>
              <div style={{marginTop:20,padding:"9px 0",textAlign:"center",borderRadius:9,background:hov===r.key?r.bc:"var(--bg3)",color:hov===r.key?"#0b0c0f":"var(--t2)",fontWeight:700,fontSize:12,transition:"all 0.3s"}}>Get Started {"\u2192"}</div>
            </div>))}
        </div>
      </div>
    </div>);
}

/* ═══════════════════════════════════════════════════════════════
   PAGE 2: PROFILE SETUP
   ═══════════════════════════════════════════════════════════════ */
function ProfileSetup({role,onComplete,onBack}){
  const is=role==="seller";
  const[zip,setZip]=useState("");const[make,setMake]=useState("Toyota");const[model,setModel]=useState("RAV4");const[year,setYear]=useState("2018");const[mileage,setMileage]=useState("45000");const[body,setBody]=useState("SUV");const[drive,setDrive]=useState("AWD");const[trim,setTrim]=useState("");const[engine,setEngine]=useState("");
  const[budget,setBudget]=useState("20000");const[prefBody,setPB]=useState("");const[prefDrive,setPD]=useState("");const[prefMake,setPM]=useState("");
  const mdls=MODELS[make]||[];
  const ok=is?(make&&model&&year&&mileage&&body&&drive):(budget&&parseFloat(budget)>0);
  const submit=()=>{if(is)onComplete({zip,make,model,year:+year,mileage:+mileage,body,drive,trim,engine});else onComplete({zip,budget:+budget,prefBody,prefDrive,prefMake});};
  return(
    <div style={{minHeight:"100vh",background:"var(--bg0)",padding:"36px 24px"}}>
      <div style={{maxWidth:680,margin:"0 auto",animation:"fadeIn 0.6s ease"}}>
        <Steps items={["Role","Your Info",is?"Estimate":"Recommendations"]}cur={1}/>
        <Btn v="g"onClick={onBack}s={{marginBottom:20,padding:"6px 0"}}>{"\u2190"} Back</Btn>
        <div style={{background:"var(--bg2)",borderRadius:18,border:"1px solid var(--bd1)",padding:32}}>
          <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:6}}>
            <span style={{fontSize:26}}>{is?"\uD83D\uDCB0":"\uD83D\uDD0D"}</span>
            <h2 style={{fontSize:22,fontWeight:800,color:"var(--t0)",letterSpacing:-0.5}}>{is?"Tell us about your car":"Set your preferences"}</h2>
          </div>
          <p style={{fontSize:12,color:"var(--t2)",marginBottom:28,lineHeight:1.6}}>{is?"Enter your vehicle details for pricing, depreciation analysis, and regional comparison.":"Set your budget and preferences to get personalized value-ranked recommendations."}</p>
          {is?(
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
              <F label="Make"><select value={make}onChange={e=>{setMake(e.target.value);setModel(MODELS[e.target.value]?.[0]||"");}}style={sel}>{MAKES.map(m=><option key={m}value={m}>{m}</option>)}</select></F>
              <F label="Model"><select value={model}onChange={e=>setModel(e.target.value)}style={sel}>{mdls.map(m=><option key={m}value={m}>{m}</option>)}<option value="__other">Other</option></select></F>
              <F label="Year"><select value={year}onChange={e=>setYear(e.target.value)}style={sel}>{YEARS.map(y=><option key={y}value={y}>{y}</option>)}</select></F>
              <F label="Mileage"><input type="number"value={mileage}onChange={e=>setMileage(e.target.value)}style={inp}placeholder="e.g. 45000"/></F>
              <F label="Body Type"><select value={body}onChange={e=>setBody(e.target.value)}style={sel}>{BODY_TYPES.map(b=><option key={b}value={b}>{b}</option>)}</select></F>
              <F label="Drive Type"><select value={drive}onChange={e=>setDrive(e.target.value)}style={sel}>{DRIVE_TYPES.map(d=><option key={d}value={d}>{d}</option>)}</select></F>
              <F label="Zip Code"hint="For regional pricing"><input type="text"value={zip}onChange={e=>setZip(e.target.value)}style={inp}placeholder="e.g. 90210"maxLength={5}/></F>
              <F label="Trim"hint="Optional"><input type="text"value={trim}onChange={e=>setTrim(e.target.value)}style={inp}placeholder="e.g. XLE"/></F>
              <F label="Engine"hint="Optional"><input type="text"value={engine}onChange={e=>setEngine(e.target.value)}style={inp}placeholder="e.g. 2.5L I4"/></F>
            </div>
          ):(
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
              <F label="Budget ($)"hint="Maximum spend"><input type="number"value={budget}onChange={e=>setBudget(e.target.value)}style={inp}placeholder="e.g. 20000"/></F>
              <F label="Zip Code"hint="Optional"><input type="text"value={zip}onChange={e=>setZip(e.target.value)}style={inp}placeholder="e.g. 90210"maxLength={5}/></F>
              <F label="Body Type"hint="Optional"><select value={prefBody}onChange={e=>setPB(e.target.value)}style={sel}><option value="">Any</option>{BODY_TYPES.map(b=><option key={b}value={b}>{b}</option>)}</select></F>
              <F label="Drive Type"hint="Optional"><select value={prefDrive}onChange={e=>setPD(e.target.value)}style={sel}><option value="">Any</option>{DRIVE_TYPES.map(d=><option key={d}value={d}>{d}</option>)}</select></F>
              <F label="Preferred Make"hint="Optional"><select value={prefMake}onChange={e=>setPM(e.target.value)}style={sel}><option value="">Any</option>{MAKES.map(m=><option key={m}value={m}>{m}</option>)}</select></F>
            </div>
          )}
          <Btn v="p"full onClick={submit}disabled={!ok}s={{marginTop:28}}>{is?"Get My Estimate \u2192":"Find Recommendations \u2192"}</Btn>
        </div>
      </div>
    </div>);
}

/* ═══════════════════════════════════════════════════════════════
   MINI SVG CHART for Depreciation Curve (Direction 3)
   ═══════════════════════════════════════════════════════════════ */
function DepreciationChart({curve,currentAge}){
  const w=560,h=200,pad={t:20,r:20,b:35,l:50};
  const iw=w-pad.l-pad.r,ih=h-pad.t-pad.b;
  const maxP=Math.max(...curve.map(c=>c.price));
  const x=age=>pad.l+(age/15)*iw;
  const y=price=>pad.t+ih-(price/maxP)*ih;
  const pts=curve.map(c=>`${x(c.age)},${y(c.price)}`).join(" ");
  const fillPts=`${x(0)},${pad.t+ih} ${pts} ${x(15)},${pad.t+ih}`;
  const cx=x(Math.min(currentAge,15));
  const cy=y(curve.find(c=>c.age===Math.min(currentAge,15))?.price||0);
  return(
    <svg viewBox={`0 0 ${w} ${h}`}style={{width:"100%",height:"auto"}}>
      <defs><linearGradient id="cg"x1="0"y1="0"x2="0"y2="1"><stop offset="0%"stopColor="var(--amber)"stopOpacity="0.25"/><stop offset="100%"stopColor="var(--amber)"stopOpacity="0"/></linearGradient></defs>
      {[0,1,2,3,4].map(i=>{const yy=pad.t+(ih/4)*i;return<g key={i}><line x1={pad.l}y1={yy}x2={w-pad.r}y2={yy}stroke="var(--bd1)"strokeWidth="1"/><text x={pad.l-8}y={yy+4}textAnchor="end"fill="var(--t3)"fontSize="9"fontFamily="JetBrains Mono">${Math.round(maxP-maxP/4*i/1000)}k</text></g>;})}
      {[0,3,5,8,10,13,15].map(a=><text key={a}x={x(a)}y={h-8}textAnchor="middle"fill="var(--t3)"fontSize="9"fontFamily="JetBrains Mono">{a}yr</text>)}
      <polygon points={fillPts}fill="url(#cg)"/>
      <polyline points={pts}fill="none"stroke="var(--amber)"strokeWidth="2.5"strokeLinejoin="round"/>
      <circle cx={cx}cy={cy}r="6"fill="var(--amber)"stroke="var(--bg2)"strokeWidth="3"/>
      <text x={cx}y={cy-14}textAnchor="middle"fill="var(--t0)"fontSize="11"fontWeight="700"fontFamily="JetBrains Mono">You</text>
    </svg>);
}

/* ═══════════════════════════════════════════════════════════════
   PAGE 3A: SELLER DASHBOARD
   ═══════════════════════════════════════════════════════════════ */
function SellerDash({profile,onBack}){
  const[result,setResult]=useState(null);const[depr,setDepr]=useState(null);const[regional,setRegional]=useState(null);const[loading,setLoading]=useState(true);const[error,setError]=useState("");
  const[simMil,setSimMil]=useState(profile.mileage);const[simYr,setSimYr]=useState(profile.year);const[simRes,setSimRes]=useState(null);
  useEffect(()=>{
    let alive=true;
    (async()=>{
      try{
        const realPrice=await fetchSellerPrice(profile);
        if(!alive)return;
        setResult(realPrice);
        setDepr(simDepreciation(profile.make,profile.model,profile.body,Math.max(2026-profile.year,0)));
        setRegional(simRegionalPrices(profile.make,profile.model,profile.year,profile.mileage,profile.body,profile.drive));
      }catch(err){
        if(!alive)return;
        setError(err.message||"Failed to load seller pricing.");
      }finally{
        if(alive)setLoading(false);
      }
    })();
    return()=>{alive=false;};
  },[profile]);

  const runSim=async()=>{
    try{
      setLoading(true);
      setError("");
      const realSim=await fetchSellerPrice({...profile,year:simYr,mileage:simMil});
      setSimRes(realSim);
    }catch(err){
      setError(err.message||"Simulation failed.");
    }finally{
      setLoading(false);
    }
  };

  if(loading&&!result)return<div style={{minHeight:"100vh",background:"var(--bg0)"}}><div style={{maxWidth:780,margin:"0 auto",padding:"36px 24px"}}><Steps items={["Role","Info","Estimate"]}cur={2}/><Spinner t="Running pricing & depreciation models..."/></div></div>;
  if(error&&!result)return<div style={{minHeight:"100vh",background:"var(--bg0)",padding:"36px 24px"}}><div style={{maxWidth:780,margin:"0 auto"}}><Steps items={["Role","Info","Estimate"]}cur={2}/><Btn v="g"onClick={onBack}s={{marginBottom:20,padding:"6px 0"}}>{"←"} Start over</Btn>
        {error&&<div style={{marginBottom:16,padding:"12px 14px",borderRadius:10,background:"#3a1717",border:"1px solid #7a2b2b",color:"#ffb4b4",fontSize:12}}>{error}</div>}<div style={{padding:20,borderRadius:14,background:"#3a1717",border:"1px solid #7a2b2b",color:"#ffb4b4"}}>{error}</div></div></div>;
  const{low,mid,high}=result;const gMax=high*1.3;const pct=v=>Math.min(v/gMax*100,98);
  return(
    <div style={{minHeight:"100vh",background:"var(--bg0)",padding:"36px 24px 80px"}}>
      <div style={{maxWidth:780,margin:"0 auto"}}>
        <Steps items={["Role","Info","Estimate"]}cur={2}/>
        <Btn v="g"onClick={onBack}s={{marginBottom:20,padding:"6px 0"}}>{"\u2190"} Start over</Btn>
        {/* Vehicle Bar */}
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"14px 20px",borderRadius:13,background:"var(--bg2)",border:"1px solid var(--bd1)",marginBottom:20,animation:"fadeIn 0.5s ease",flexWrap:"wrap",gap:10}}>
          <div style={{display:"flex",alignItems:"center",gap:10}}><span style={{fontSize:26}}>{"\uD83D\uDE97"}</span><div><div style={{fontSize:17,fontWeight:800,color:"var(--t0)"}}>{profile.year} {profile.make} {profile.model}</div><div style={{fontSize:11,color:"var(--t2)"}}>{profile.mileage.toLocaleString()} mi · {profile.body} · {profile.drive}{profile.trim?` · ${profile.trim}`:""}</div></div></div>
          <div style={{display:"flex",gap:6}}><Badge t={TIER[profile.make]||"Economy"}c={TIER[profile.make]==="Luxury"?"var(--amber)":TIER[profile.make]==="Mid"?"var(--blue)":"var(--t2)"}/><Badge t={depr.holdValueRank}c={depr.holdValueRank==="Good"?"var(--green)":"var(--amber)"}/></div>
        </div>

        {/* DIR 1: Price Cards */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14,marginBottom:20,animation:"slideUp 0.5s ease 0.1s both"}}>
          {[{l:"Competitive",v:low,c:"var(--green)",i:"\u26A1",s:"Sells quickly"},{l:"Fair Market",v:mid,c:"var(--amber)",i:"\u2696\uFE0F",s:"Balanced estimate"},{l:"Premium",v:high,c:"var(--red)",i:"\uD83D\uDC8E",s:"May sit longer"}].map(({l,v,c,i,s})=>(
            <div key={l}style={{padding:20,borderRadius:14,background:"var(--bg2)",border:"1px solid var(--bd1)",textAlign:"center"}}>
              <div style={{fontSize:22,marginBottom:6}}>{i}</div>
              <div style={{fontSize:9,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1.5,marginBottom:6,fontFamily:"'JetBrains Mono',monospace"}}>{l}</div>
              <div style={{fontSize:28,fontWeight:900,color:c,fontFamily:"'JetBrains Mono',monospace",letterSpacing:-1}}>${v.toLocaleString()}</div>
              <div style={{fontSize:10,color:"var(--t3)",marginTop:4}}>{s}</div></div>))}
        </div>

        {/* DIR 1: Gauge */}
        <Section title="Price Spectrum"delay={0.15}>
          <div style={{position:"relative",height:12,borderRadius:6,overflow:"hidden",background:"linear-gradient(90deg, var(--green) 0%, var(--amber) 50%, var(--red) 100%)"}}><div style={{position:"absolute",inset:0,background:"rgba(0,0,0,0.3)"}}/></div>
          <div style={{position:"relative",height:65,marginTop:4}}>
            {[{v:low,l:"Competitive",c:"var(--green)"},{v:mid,l:"Fair",c:"var(--amber)"},{v:high,l:"Premium",c:"var(--red)"}].map(({v,l,c})=>(
              <div key={l}style={{position:"absolute",left:`${pct(v)}%`,top:0,transform:"translateX(-50%)",textAlign:"center"}}>
                <div style={{width:2,height:16,background:c,margin:"0 auto"}}/>
                <div style={{fontSize:12,fontWeight:800,color:c,fontFamily:"'JetBrains Mono',monospace",marginTop:2}}>${v.toLocaleString()}</div>
                <div style={{fontSize:8,color:"var(--t3)",fontWeight:600,textTransform:"uppercase",letterSpacing:1,marginTop:2}}>{l}</div></div>))}
          </div>
        </Section>

        {/* DIR 3: Depreciation Stage */}
        <Section title="Depreciation Analysis" icon={"\uD83D\uDCC9"} delay={0.2}>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginBottom:18}}>
            <div style={{padding:16,borderRadius:12,background:"var(--bg3)",border:`2px solid ${depr.stageColor}40`}}>
              <div style={{fontSize:9,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1.5,marginBottom:6}}>Current Stage</div>
              <div style={{fontSize:18,fontWeight:800,color:depr.stageColor}}>{depr.stage}</div>
              <p style={{fontSize:11,color:"var(--t2)",marginTop:6,lineHeight:1.5}}>{depr.stageDesc}</p>
            </div>
            <div style={{padding:16,borderRadius:12,background:"var(--bg3)",border:"1px solid var(--bd1)"}}>
              <div style={{fontSize:9,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1.5,marginBottom:6}}>If You Keep 2 More Years</div>
              <div style={{fontSize:22,fontWeight:800,color:"var(--red)",fontFamily:"'JetBrains Mono',monospace"}}>-${depr.lossIn2Years.toLocaleString()}</div>
              <div style={{fontSize:11,color:"var(--t2)",marginTop:4}}>Estimated {depr.lossPct}% value loss</div>
              <div style={{fontSize:11,color:"var(--t2)",marginTop:2}}>Annual depreciation rate: ~{depr.rate}%</div>
            </div>
          </div>
          <div style={{fontSize:11,fontWeight:600,color:"var(--t1)",marginBottom:10}}>Estimated Value Over Time — {profile.make} {profile.model}</div>
          <DepreciationChart curve={depr.curve}currentAge={Math.max(2026-profile.year,0)}/>
        </Section>

        {/* DIR 3: Regional Price Comparison */}
        <Section title="Regional Price Comparison" icon={"\uD83D\uDDFA\uFE0F"} delay={0.3}>
          <p style={{fontSize:11,color:"var(--t3)",marginBottom:14}}>Same vehicle, different regions — controlling for make, model, year, and mileage.</p>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
            {regional.map((r,i)=>(
              <div key={r.region}style={{padding:"10px 14px",borderRadius:10,background:"var(--bg3)",border:`1px solid ${i===0?"var(--green)30":"var(--bd1)"}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                <div><div style={{fontSize:12,fontWeight:700,color:"var(--t0)"}}>{r.region}</div>{i===0&&<div style={{fontSize:9,color:"var(--green)",fontWeight:700,marginTop:2}}>CHEAPEST</div>}</div>
                <div style={{textAlign:"right"}}><div style={{fontSize:14,fontWeight:800,color:r.diff<0?"var(--green)":r.diff>0?"var(--red)":"var(--t1)",fontFamily:"'JetBrains Mono',monospace"}}>${r.price.toLocaleString()}</div><div style={{fontSize:9,color:"var(--t3)"}}>{r.diff>0?"+":""}{r.diff}% vs avg</div></div>
              </div>))}
          </div>
        </Section>

        {/* DIR 1: What-If */}
        <Section title="What-If Simulator" icon={"\uD83D\uDD2E"} delay={0.35}>
          <p style={{fontSize:11,color:"var(--t3)",marginBottom:14}}>Adjust mileage or year to see how the price estimate changes.</p>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr auto",gap:14,alignItems:"end"}}>
            <F label="Mileage"><input type="number"value={simMil}onChange={e=>setSimMil(+e.target.value||0)}style={inp}/></F>
            <F label="Year"><select value={simYr}onChange={e=>setSimYr(+e.target.value)}style={sel}>{YEARS.map(y=><option key={y}value={y}>{y}</option>)}</select></F>
            <Btn v="p"onClick={runSim}>Simulate</Btn>
          </div>
          {simRes&&<div style={{marginTop:16,display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10}}>
            {[{l:"Competitive",o:low,s:simRes.low,c:"var(--green)"},{l:"Fair",o:mid,s:simRes.mid,c:"var(--amber)"},{l:"Premium",o:high,s:simRes.high,c:"var(--red)"}].map(({l,o,s,c})=>{const d=s-o;return(
              <div key={l}style={{padding:14,borderRadius:10,background:"var(--bg3)",border:"1px solid var(--bd1)",textAlign:"center"}}>
                <div style={{fontSize:9,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1.5,marginBottom:5}}>{l}</div>
                <div style={{fontSize:18,fontWeight:800,color:c,fontFamily:"'JetBrains Mono',monospace"}}>${s.toLocaleString()}</div>
                <div style={{fontSize:10,fontWeight:700,marginTop:3,color:d>0?"var(--green)":d<0?"var(--red)":"var(--t3)",fontFamily:"'JetBrains Mono',monospace"}}>{d>0?"+":""}{d!==0?`$${d.toLocaleString()}`:"—"}</div></div>);})}</div>}
        </Section>

        {/* Tips */}
        <Section title="Pricing Tips" icon={"\uD83D\uDCA1"} delay={0.4}>
          <div style={{display:"flex",flexDirection:"column",gap:10}}>
            {[{t:"Start at Fair Market",d:`List around $${mid.toLocaleString()} to attract serious buyers while leaving negotiation room.`},{t:"Competitive for Speed",d:`Price at $${low.toLocaleString()} for faster interest and quicker sale.`},{t:depr.stage==="Steep Drop"?"Consider Selling Soon":"Timing Flexibility",d:depr.stage==="Steep Drop"?`Your car is depreciating fast (~${depr.rate}%/yr). Selling sooner preserves more value.`:`Your car's depreciation has stabilized. You have more flexibility on timing.`},{t:"Regional Advantage",d:`The cheapest region for this car type is ${regional[0]?.region}. If buyers compare, price competitively.`}].map(({t,d})=>(
              <div key={t}style={{padding:"12px 14px",borderRadius:9,background:"var(--bg3)",border:"1px solid var(--bd1)"}}><div style={{fontSize:12,fontWeight:700,color:"var(--t0)",marginBottom:3}}>{t}</div><div style={{fontSize:11,color:"var(--t2)",lineHeight:1.6}}>{d}</div></div>))}
          </div>
        </Section>
      </div>
    </div>);
}

/* ═══════════════════════════════════════════════════════════════
   PAGE 3B: BUYER DASHBOARD
   ═══════════════════════════════════════════════════════════════ */
function BuyerDash({profile,onBack}){
  const[results,setResults]=useState(null);const[loading,setLoading]=useState(true);
  const[maxMil,setMaxMil]=useState("");const[minYr,setMinYr]=useState("");
  const[fMake,setFMake]=useState(profile.prefMake||"");const[fBody,setFBody]=useState(profile.prefBody||"");const[fDrive,setFDrive]=useState(profile.prefDrive||"");const[budget,setBudget]=useState(profile.budget);
  const[cmpSet,setCmpSet]=useState(new Set());
  const[error,setError]=useState("");
  const run=useCallback(async (b,fb,fd,fm,mm,my)=>{
    setLoading(true);
    setError("");
    try{
      const realResults = await fetchBuyerRecommendations({
        budget:b,
        body_type:fb||null,
        drive_type:fd||null,
        make:fm||null,
        max_mileage:mm?+mm:null,
        min_year:my?+my:null,
        zipcode:profile.zip||null,
        top_n:10,
      });
      setResults(realResults);
      setCmpSet(new Set());
    }catch(err){
      setResults([]);
      setError(err?.message || "Failed to load recommendations");
    }finally{
      setLoading(false);
    }
  },[profile.zip]);
  useEffect(()=>{run(profile.budget,profile.prefBody,profile.prefDrive,profile.prefMake,null,null);},[run, profile.budget, profile.prefBody, profile.prefDrive, profile.prefMake]);
  const togCmp=t=>setCmpSet(p=>{const n=new Set(p);if(n.has(t))n.delete(t);else if(n.size<3)n.add(t);return n;});
  const cmpd=results?results.filter(r=>cmpSet.has(r.title)):[];

  // Direction 3: Regional insight for buyer
  const topPick=results&&results.length>0?results[0]:null;
  const buyerRegional=topPick?simRegionalPrices(topPick.make,topPick.model,topPick.typical_year,topPick.typical_mileage,topPick.body_type,topPick.drive_type):[];

  return(
    <div style={{minHeight:"100vh",background:"var(--bg0)",padding:"36px 24px 80px"}}>
      <div style={{maxWidth:900,margin:"0 auto"}}>
        <Steps items={["Role","Info","Recommendations"]}cur={2}/>
        <Btn v="g"onClick={onBack}s={{marginBottom:20,padding:"6px 0"}}>{"\u2190"} Start over</Btn>
        {/* Budget Banner */}
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"14px 20px",borderRadius:13,background:"var(--bg2)",border:"1px solid var(--bd1)",marginBottom:20,animation:"fadeIn 0.5s ease",flexWrap:"wrap",gap:10}}>
          <div><div style={{fontSize:10,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'JetBrains Mono',monospace"}}>Your Budget</div><div style={{fontSize:24,fontWeight:900,color:"var(--green)",fontFamily:"'JetBrains Mono',monospace",letterSpacing:-1}}>${budget.toLocaleString()}</div></div>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>{fBody&&<Badge t={fBody}c="var(--blue)"/>}{fDrive&&<Badge t={fDrive}c="var(--amber)"/>}{fMake&&<Badge t={fMake}c="var(--green)"/>}</div>
        </div>
        {/* DIR 2: Filter Panel */}
        <Section title="Refine Your Search" icon={"\uD83D\uDD27"} delay={0.05}>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14}}>
            <F label="Budget ($)"><input type="number"value={budget}onChange={e=>setBudget(+e.target.value||0)}style={inp}/></F>
            <F label="Body Type"><select value={fBody}onChange={e=>setFBody(e.target.value)}style={sel}><option value="">Any</option>{BODY_TYPES.map(b=><option key={b}value={b}>{b}</option>)}</select></F>
            <F label="Drive Type"><select value={fDrive}onChange={e=>setFDrive(e.target.value)}style={sel}><option value="">Any</option>{DRIVE_TYPES.map(d=><option key={d}value={d}>{d}</option>)}</select></F>
            <F label="Preferred Make"><select value={fMake}onChange={e=>setFMake(e.target.value)}style={sel}><option value="">Any</option>{MAKES.map(m=><option key={m}value={m}>{m}</option>)}</select></F>
            <F label="Max Mileage"hint="Blank = no limit"><input type="number"value={maxMil}onChange={e=>setMaxMil(e.target.value)}style={inp}placeholder="e.g. 80000"/></F>
            <F label="Min Year"hint="Blank = no limit"><input type="number"value={minYr}onChange={e=>setMinYr(e.target.value)}style={inp}placeholder="e.g. 2015"/></F>
          </div>
          <Btn v="p"full onClick={()=>run(budget,fBody,fDrive,fMake,maxMil,minYr)}s={{marginTop:16}}>Update Results</Btn>
        </Section>

        {loading&&<Spinner t="Searching for best value cars..."/>}
        {!loading&&results&&results.length===0&&(
          <Section title="No matches found" icon={"⚠️"} delay={0}>
            <p style={{fontSize:12,color:"var(--t2)",lineHeight:1.7}}>Try increasing your budget, relaxing mileage/year filters, or removing the preferred make/body constraints.</p>
          </Section>
        )}
        {!loading&&results&&results.length>0&&(<>
          {results.length===0?(
            <div style={{textAlign:"center",padding:56,background:"var(--bg2)",borderRadius:16,border:"1px solid var(--bd1)"}}><div style={{fontSize:38,marginBottom:10}}>{"\uD83D\uDD0D"}</div><h3 style={{fontSize:17,fontWeight:700,color:"var(--t0)",marginBottom:6}}>No matches found</h3><p style={{color:"var(--t2)",fontSize:12}}>Try increasing your budget or loosening filters.</p></div>
          ):(<>
            {/* DIR 2: Top Pick */}
            <div style={{padding:"18px 22px",borderRadius:13,marginBottom:18,background:"linear-gradient(135deg, var(--greenDim) 0%, var(--blueDim) 100%)",border:"1px solid #2fc87230",animation:"scaleIn 0.4s ease"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:8}}>
                <div><div style={{fontSize:9,fontWeight:700,color:"var(--green)",letterSpacing:1.5,textTransform:"uppercase",marginBottom:3,fontFamily:"'JetBrains Mono',monospace"}}>{"\u2605"} Top Pick</div><span style={{fontSize:18,fontWeight:800,color:"var(--t0)"}}>{results[0].title}</span><span style={{fontSize:12,color:"var(--t2)",marginLeft:8}}>({results[0].year_range})</span><div style={{display:"flex",gap:6,marginTop:6}}><Badge t={results[0].dealLabel}c={results[0].dealColor}/><Badge t={results[0].deprLabel}c={results[0].deprColor}/></div></div>
                <div style={{textAlign:"right"}}><div style={{fontSize:17,fontWeight:800,color:"var(--green)",fontFamily:"'JetBrains Mono',monospace"}}>${results[0].typical_price.toLocaleString()}</div><div style={{fontSize:10,color:"var(--t3)"}}>Fair value: ${results[0].predicted_fair.toLocaleString()}</div></div>
              </div>
            </div>
            {/* Stats */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:20,animation:"slideUp 0.4s ease 0.1s both"}}>
              {[{l:"Results",v:results.length,c:"var(--t0)"},{l:"Avg Price",v:`$${Math.round(results.reduce((a,r)=>a+r.typical_price,0)/results.length).toLocaleString()}`,c:"var(--amber)"},{l:"Best Value",v:`${Math.max(...results.map(r=>r.avg_value_pct)).toFixed(0)}%`,c:"var(--green)"},{l:"Great Deals",v:results.filter(r=>r.dealLabel==="Great Deal").length,c:"var(--blue)"}].map(({l,v,c})=>(
                <div key={l}style={{padding:"12px 14px",borderRadius:11,background:"var(--bg2)",border:"1px solid var(--bd1)",textAlign:"center"}}><div style={{fontSize:8,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1.5,marginBottom:3,fontFamily:"'JetBrains Mono',monospace"}}>{l}</div><div style={{fontSize:16,fontWeight:800,color:c,fontFamily:"'JetBrains Mono',monospace"}}>{v}</div></div>))}
            </div>
            {/* DIR 2: Recommendation Cards */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginBottom:24}}>
              {results.map((rec,i)=>{const ic=cmpSet.has(rec.title);return(
                <div key={`${rec.title}-${i}`}style={{padding:20,borderRadius:14,background:"var(--bg2)",border:`1px solid ${ic?"var(--amber)":"var(--bd1)"}`,animation:`slideUp 0.4s ease ${0.04*i}s both`,transition:"border-color 0.2s"}}>
                  {/* Car Image */}
                  {rec.image_url ? (
                    <div style={{marginBottom:12,borderRadius:10,overflow:"hidden",height:140,background:"var(--bg3)"}}>
                      <img 
                        src={rec.image_url} 
                        alt={rec.title}
                        style={{width:"100%",height:"100%",objectFit:"cover"}}
                        onError={(e)=>{e.target.style.display='none';e.target.parentElement.style.display='none';}}
                      />
                    </div>
                  ) : (
                    <div style={{marginBottom:12,borderRadius:10,height:140,background:"var(--bg3)",display:"flex",alignItems:"center",justifyContent:"center",border:"1px dashed var(--bd1)"}}>
                      <span style={{fontSize:40}}>🚗</span>
                    </div>
                  )}
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:12}}>
                    <div><div style={{fontSize:9,fontWeight:700,color:"var(--amber)",letterSpacing:1.5,textTransform:"uppercase",marginBottom:3,fontFamily:"'JetBrains Mono',monospace"}}>#{i+1}</div><h4 style={{fontSize:16,fontWeight:800,color:"var(--t0)",margin:0}}>{rec.title}</h4></div>
                    <Badge t={rec.confidence}c={rec.confidence==="High"?"var(--green)":rec.confidence==="Medium"?"var(--amber)":"var(--t3)"}/>
                  </div>
                  {/* Deal + Depreciation badges */}
                  <div style={{display:"flex",gap:5,marginBottom:12}}><Badge t={rec.dealLabel}c={rec.dealColor}/><Badge t={rec.deprLabel}c={rec.deprColor}/></div>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:7,marginBottom:12}}>
                    {[["Years",rec.year_range],["Mileage",`${rec.typical_mileage.toLocaleString()} mi`],["Body",rec.body_type],["Drive",rec.drive_type]].map(([l,v])=>(
                      <div key={l}style={{padding:"7px 9px",borderRadius:7,background:"var(--bg3)",border:"1px solid var(--bd1)"}}><div style={{fontSize:8,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1}}>{l}</div><div style={{fontSize:12,fontWeight:700,color:"var(--t0)",fontFamily:"'JetBrains Mono',monospace"}}>{v}</div></div>))}
                  </div>
                  <div style={{display:"flex",gap:14,padding:"10px 0",borderTop:"1px solid var(--bd1)"}}>
                    <div style={{flex:1}}><div style={{fontSize:8,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1}}>Price</div><div style={{fontSize:18,fontWeight:800,color:"var(--t0)",fontFamily:"'JetBrains Mono',monospace"}}>${rec.typical_price.toLocaleString()}</div></div>
                    <div style={{flex:1}}><div style={{fontSize:8,fontWeight:700,color:"var(--t3)",textTransform:"uppercase",letterSpacing:1}}>Fair Value</div><div style={{fontSize:18,fontWeight:800,color:"var(--green)",fontFamily:"'JetBrains Mono',monospace"}}>${rec.predicted_fair.toLocaleString()}</div></div>
                  </div>
                  {rec.avg_value_pct>0&&<div style={{padding:"5px 9px",borderRadius:7,background:"var(--greenDim)",border:"1px solid #2fc87225",fontSize:10,color:"var(--green)",fontWeight:700,fontFamily:"'JetBrains Mono',monospace",marginTop:3}}>{"\u2193"} {rec.avg_value_pct.toFixed(1)}% below fair value</div>}
                  <p style={{fontSize:10,color:"var(--t3)",lineHeight:1.6,margin:"8px 0 0",fontStyle:"italic"}}>{rec.reason}</p>
                  <button onClick={()=>togCmp(rec.title)}style={{marginTop:10,width:"100%",padding:"7px",borderRadius:7,fontSize:10,fontWeight:700,cursor:"pointer",transition:"all 0.2s",fontFamily:"'Outfit',sans-serif",background:ic?"var(--amberDim)":"var(--bg3)",color:ic?"var(--amber)":"var(--t2)",border:`1px solid ${ic?"var(--amberMid)":"var(--bd1)"}`}}>{ic?"\u2713 In comparison":"Add to compare"}</button>
                </div>);})}
            </div>

            {/* DIR 3: Regional Insight for Buyers */}
            {topPick&&buyerRegional.length>0&&(
              <Section title={`Regional Prices — ${topPick.title}`} icon={"\uD83D\uDDFA\uFE0F"} delay={0.1}>
                <p style={{fontSize:11,color:"var(--t3)",marginBottom:12}}>Where this model tends to be cheaper or more expensive.</p>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:8}}>
                  {buyerRegional.slice(0,8).map((r,i)=>(
                    <div key={r.region}style={{padding:"9px 10px",borderRadius:9,background:"var(--bg3)",border:`1px solid ${i===0?"var(--green)30":"var(--bd1)"}`,textAlign:"center"}}>
                      <div style={{fontSize:10,fontWeight:700,color:"var(--t1)",marginBottom:3}}>{r.region}</div>
                      <div style={{fontSize:13,fontWeight:800,color:r.diff<0?"var(--green)":r.diff>0?"var(--red)":"var(--t1)",fontFamily:"'JetBrains Mono',monospace"}}>${r.price.toLocaleString()}</div>
                      <div style={{fontSize:8,color:"var(--t3)",marginTop:2}}>{r.diff>0?"+":""}{r.diff}%{i===0?" · Cheapest":""}</div>
                    </div>))}
                </div>
              </Section>
            )}

            {/* DIR 2: Compare Panel */}
            {cmpd.length>=2&&(
              <Section title={`Side-by-Side Comparison (${cmpd.length})`} icon={"\u2696\uFE0F"} delay={0}>
                <div style={{overflowX:"auto"}}>
                  <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                    <thead><tr><th style={{textAlign:"left",padding:"8px 10px",borderBottom:"1px solid var(--bd2)",color:"var(--t3)",fontSize:9,fontWeight:700,textTransform:"uppercase",letterSpacing:1.5}}>Metric</th>{cmpd.map(c=><th key={c.title}style={{textAlign:"center",padding:"8px 10px",borderBottom:"1px solid var(--bd2)",color:"var(--t0)",fontWeight:700,fontSize:13}}>{c.title}</th>)}</tr></thead>
                    <tbody>{[["Price",c=>`$${c.typical_price.toLocaleString()}`],["Fair Value",c=>`$${c.predicted_fair.toLocaleString()}`],["Deal",c=>c.dealLabel],["Value Gap",c=>`${c.avg_value_pct.toFixed(1)}%`],["Years",c=>c.year_range],["Mileage",c=>`${c.typical_mileage.toLocaleString()} mi`],["Depreciation",c=>c.deprLabel],["Confidence",c=>c.confidence],["Samples",c=>c.sample_count]].map(([m,fn])=>(
                      <tr key={m}><td style={{padding:"8px 10px",borderBottom:"1px solid var(--bd1)",color:"var(--t2)",fontWeight:600,fontSize:11}}>{m}</td>{cmpd.map(c=><td key={c.title}style={{padding:"8px 10px",textAlign:"center",borderBottom:"1px solid var(--bd1)",color:"var(--t0)",fontFamily:"'JetBrains Mono',monospace",fontWeight:600}}>{fn(c)}</td>)}</tr>))}</tbody>
                  </table>
                </div>
              </Section>
            )}
          </>)}
        </>)}
      </div>
    </div>);
}

/* ═══════════════════════════════════════════════════════════════
   ROOT
   ═══════════════════════════════════════════════════════════════ */
export default function App(){
  const[page,setPage]=useState("role");const[role,setRole]=useState(null);const[profile,setProfile]=useState(null);
  const go=(p,r,pr)=>{setPage(p);if(r!==undefined)setRole(r);if(pr!==undefined)setProfile(pr);};
  return(<>
    <style>{CSS}</style>
    <div style={{fontFamily:"'Outfit',sans-serif",background:"var(--bg0)",color:"var(--t0)",minHeight:"100vh"}}>
      {page==="role"&&<RoleSelect onSelect={r=>go("profile",r)}/>}
      {page==="profile"&&<ProfileSetup role={role}onComplete={p=>go("dash",undefined,p)}onBack={()=>go("role",null,null)}/>}
      {page==="dash"&&role==="seller"&&<SellerDash profile={profile}onBack={()=>go("role",null,null)}/>}
      {page==="dash"&&role==="buyer"&&<BuyerDash profile={profile}onBack={()=>go("role",null,null)}/>}
    </div>
  </>);
}
