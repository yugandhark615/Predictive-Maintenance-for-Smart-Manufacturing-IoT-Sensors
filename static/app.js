function data(id){const el=document.getElementById(id);return el?JSON.parse(el.textContent):null}

const LEVEL_COLOR={Low:"#35C48C",Medium:"#E8A33D",High:"#E5484D"};

Chart.defaults.color="#8A96A3";
Chart.defaults.font.family="'IBM Plex Sans', sans-serif";
Chart.defaults.borderColor="#2A3542";

const risks=data("read")||{};
const imp=data("imp");
const featureStats=data("fstats");
const series=data("series");

// Sensor Reading Profile -- shared by both modes.
const names=Object.keys(risks);
if(names.length){
  new Chart(document.getElementById("readings"),{
    type:"bar",
    data:{labels:names,datasets:[{label:"Reading %",data:names.map(n=>risks[n].pct),backgroundColor:names.map(n=>LEVEL_COLOR[risks[n].level]),borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>v+"%"},grid:{color:"#2A3542"}},x:{grid:{display:false}}}}
  });
}

// Manual mode -- ML feature importance.
if(imp){
  new Chart(document.getElementById("importance"),{
    type:"bar",
    data:{labels:imp.map(x=>x[0]),datasets:[{data:imp.map(x=>x[1]),backgroundColor:"#4FA8E0",borderRadius:4}]},
    options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:"#2A3542"}},y:{grid:{display:false}}}}
  });
}

// Dataset mode -- threshold breach rate + per-sensor trend with limit line.
if(featureStats){
  new Chart(document.getElementById("breach"),{
    type:"bar",
    data:{labels:featureStats.map(f=>f.label),datasets:[{data:featureStats.map(f=>f.breach_pct),backgroundColor:featureStats.map(f=>f.breach_count?"#E5484D":"#35C48C"),borderRadius:4}]},
    options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{callback:v=>v+"%"},grid:{color:"#2A3542"}},y:{grid:{display:false}}}}
  });

  featureStats.forEach(f=>{
    const canvas=document.getElementById("trend_"+f.key.replace(/[^a-zA-Z0-9]/g,""));
    if(!canvas)return;
    const vals=(series[f.key]||[]).map(Number);
    new Chart(canvas,{
      type:"line",
      data:{labels:vals.map((_,i)=>i+1),datasets:[
        {label:f.label,data:vals,borderColor:"#4FA8E0",backgroundColor:"transparent",borderWidth:1.5,tension:.15,
         pointRadius:vals.map(v=>v>f.threshold?3:0),pointBackgroundColor:"#E5484D",pointBorderColor:"#E5484D"},
        {label:"Limit",data:vals.map(()=>f.threshold),borderColor:"#E8A33D",borderDash:[5,4],borderWidth:1.5,pointRadius:0}
      ]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:"#2A3542"}}}}
    });
  });
}
