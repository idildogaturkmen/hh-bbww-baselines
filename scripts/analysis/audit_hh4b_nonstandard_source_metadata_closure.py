#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,subprocess,tarfile
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Iterable

TEXT_EXT={".cfg",".conf",".csv",".dat",".ini",".jdl",".json",".log",".md",".py",".sh",".sub",".toml",".tsv",".txt",".yaml",".yml"}
FORBIDDEN=(".root",".parquet",".hepmc",".lhe",".lhe.gz",".tar",".tar.gz",".tgz",".zip",".xz",".bz2")
PRUNE={".cache",".conda",".git",".ipynb_checkpoints",".local","__pycache__","node_modules","venv","venvs"}
MAX_BYTES=8*1024*1024
QCD_RE=re.compile(
    r"(?P<tag>qcd_hardqcd_[A-Za-z0-9_]+?_bin(?P<bin>\d+)_"
    r"pthat(?P<low>\d+)to(?P<high>\d+|inf(?:inity)?)_"
    r"shard(?P<shard>\d+)_seed(?P<seed>\d+))",
    flags=re.IGNORECASE,
)
GGF_RE=re.compile(r"(?P<tag>HH4b_ggf_[A-Za-z0-9_]+?_shard(?P<shard>\d+)_seed(?P<seed>\d+))")
KEY_RE=re.compile(r"(n_events|generated_events|sum_event_weights|sum_weights|sum_squared|event_weight|min_event|max_event|pthat|sigma_gen|cross_section|xsec|proposal|minimum_fraction|sampling|seed|shard|tag|beam|energy|decay|higgs)",re.I)
CFG_RE=re.compile(r"(HardQCD:all|PhaseSpace:pTHat|Beams:eCM|Higgs|25:on|35:on|pythia)",re.I)

def read_tsv(p:Path):
    with p.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h,delimiter="\t"))
def write_tsv(p:Path,rows:list[dict],fields:list[str]):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,delimiter="\t",lineterminator="\n",extrasaction="ignore");w.writeheader();w.writerows(rows)
def sha_file(p:Path):
    d=hashlib.sha256()
    with p.open("rb") as h:
        for b in iter(lambda:h.read(8*1024*1024),b""):d.update(b)
    return d.hexdigest()
def sha_bytes(b:bytes):return hashlib.sha256(b).hexdigest()
def flatten(x:Any,p="")->Iterable[tuple[str,Any]]:
    if isinstance(x,dict):
        for k,v in x.items():yield from flatten(v,f"{p}.{k}" if p else str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from flatten(v,f"{p}[{i}]")
    else:yield p,x
def key(r):return (r["sample_class"],r["process_or_mode"],r["campaign"])

def parse_pthat_upper_bound(raw:str):
    normalized=str(raw).strip().lower()
    if normalized in {"inf","infinity"}:
        return "Inf"
    value=int(raw)
    if value<0:
        raise ValueError(f"negative pTHat upper bound: {raw!r}")
    return value

def pthat_interval_valid(low,high):
    low_value=int(low)
    if low_value<0:
        return False
    if str(high).strip().lower() in {"inf","infinity"}:
        return True
    return low_value<int(high)

def qcd_group_sort_key(item):
    campaign,bin_index,low,high=item[0]
    high_value=(
        float("inf")
        if str(high).strip().lower() in {"inf","infinity"}
        else int(high)
    )
    return (campaign,int(bin_index),int(low),high_value)

def parse_tag(text:str,process:str):
    m=(QCD_RE if process=="qcd_hardqcd" else GGF_RE).search(text)
    if not m:return {"source_tag":"","bin_index":"","pthat_min_GeV":"","pthat_max_GeV":"","shard_id":"","seed":""}
    g=m.groupdict()
    return {
      "source_tag":g["tag"],"bin_index":int(g["bin"]) if "bin" in g else "",
      "pthat_min_GeV":int(g["low"]) if "low" in g else "",
      "pthat_max_GeV":parse_pthat_upper_bound(g["high"]) if "high" in g else "",
      "shard_id":int(g["shard"]),"seed":int(g["seed"])
    }
def safe_text(p:Path):
    n=p.name.lower()
    if any(n.endswith(s) for s in FORBIDDEN) or p.suffix.lower() not in TEXT_EXT:return False
    try:s=p.stat().st_size
    except OSError:return False
    return 0<s<=MAX_BYTES
def text_read(p:Path):
    try:b=p.read_bytes()
    except OSError:return None
    if b"\0" in b:return None
    try:return b.decode()
    except UnicodeDecodeError:return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",type=Path,required=True)
    ap.add_argument("--blockers",type=Path,required=True)
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--eos-host",required=True)
    ap.add_argument("--search-root",type=Path,action="append",default=[])
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--source-commit",required=True)
    a=ap.parse_args()

    inv=read_tsv(a.inventory); blockers=read_tsv(a.blockers); prior=read_tsv(a.prior); manifest=read_tsv(a.manifest)
    blocker_keys={key(r) for r in blockers}
    campaigns=[r for r in inv if key(r) in blocker_keys]
    if len(campaigns)!=6 or len(blockers)!=6 or len(prior)!=6:raise RuntimeError("expected six nonstandard campaigns")
    if sum(r["process_or_mode"]=="ggf_hh4b" for r in campaigns)!=2:raise RuntimeError("expected two ggF campaigns")
    if sum(r["process_or_mode"]=="qcd_hardqcd" for r in campaigns)!=4:raise RuntimeError("expected four QCD campaigns")
    expected_bundles=sum(int(r["members"]) for r in campaigns)
    expected_events=sum(int(r["generated_events"]) for r in campaigns)
    if expected_bundles!=361 or expected_events!=2610000:raise RuntimeError(f"unexpected totals {expected_bundles}/{expected_events}")

    a.output.mkdir(parents=True, exist_ok=True)
    listing_dir=a.output/"eos_listings";listing_dir.mkdir()
    bundle_rows=[];by_campaign=defaultdict(list)
    blocker_map={key(r):r for r in blockers}
    registry=[]
    for r in sorted(campaigns,key=lambda x:(x["process_or_mode"],x["campaign"])):
        k=key(r); directory=str(Path(r["source_locator_example"]).parent)
        result=subprocess.run(["xrdfs",a.eos_host,"ls","-l",directory],check=True,text=True,stdout=subprocess.PIPE)
        paths=[line.split()[-1] for line in result.stdout.splitlines() if line.strip() and line.split()[-1].endswith("_bundle.tar.gz")]
        if len(paths)!=int(r["members"]):raise RuntimeError(f"EOS count {k}: {len(paths)} != {r['members']}")
        reg={**{f:r[f] for f in ("sample_class","process_or_mode","campaign","members","generated_events","train_members","validation_members","final_evaluation_members","source_locator_example")},"blocking_reason":blocker_map[k]["blocking_reason"],"eos_directory":directory}
        registry.append(reg)
        rows=[]
        for path in sorted(paths):
            parsed=parse_tag(Path(path).name,r["process_or_mode"])
            row={"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"remote_bundle_path":path,"bundle_basename":Path(path).name,**parsed,"exact_metadata_files":0,"structured_metadata_rows":0,"status":"eos_bundle_identity_frozen"}
            rows.append(row);bundle_rows.append(row);by_campaign[k].append(row)
        write_tsv(listing_dir/f"{r['process_or_mode']}__{r['campaign']}.tsv",rows,list(rows[0]))
    write_tsv(a.output/"nonstandard_campaign_registry.tsv",registry,list(registry[0]))

    # Reverify representative provenance and collect structured/config evidence.
    manifest_map={r["archive_path"]:r for r in manifest}
    rep_files=[]; structured=[]; config=[]
    with tarfile.open(a.archive,"r:gz") as tar:
        members={m.name:m for m in tar.getmembers()}
        if set(members)!=set(manifest_map):raise RuntimeError("archive/manifest mismatch")
        for name,m in sorted(members.items()):
            meta=manifest_map[name];k=(meta["sample_class"],meta["process_or_mode"],meta["campaign"])
            if k not in blocker_keys:continue
            h=tar.extractfile(m)
            if h is None:raise RuntimeError(name)
            b=h.read()
            if len(b)!=int(meta["size_bytes"]) or sha_bytes(b)!=meta["sha256"]:raise RuntimeError("provenance checksum mismatch")
            txt=b.decode(); parsed=parse_tag(meta["member_name"]+"\n"+txt[:20000],k[1])
            rep_files.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"source_path":name,"source_sha256":meta["sha256"],"source_size_bytes":len(b),**parsed,"status":"representative_provenance_reverified"})
            if meta["member_name"].lower().endswith(".json"):
                try:obj=json.loads(txt)
                except json.JSONDecodeError:obj=None
                if obj is not None:
                    for field,value in flatten(obj):
                        if KEY_RE.search(field):
                            structured.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"source_scope":"representative_provenance","source_path":name,"source_tag":parsed["source_tag"],"field":field,"value":value,"status":"structured_metadata_candidate"})
            for n,line in enumerate(txt.splitlines(),1):
                if CFG_RE.search(line):
                    config.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"source_scope":"representative_provenance","source_path":name,"source_line":n,"line_text":" ".join(line.split())[:1000],"status":"configuration_evidence_candidate"})

    # Targeted local text search. Only paths containing one of the six campaign names are opened.
    local_files=[]; roots=[]; seen=set()
    campaign_names=[r["campaign"] for r in campaigns]
    for root in a.search_root:
        root=root.resolve(); stats={"search_root":str(root),"exists":root.exists(),"files_considered":0,"text_files_opened":0,"campaign_matched_files":0,"json_files_parsed":0,"errors":0,"status":"approved_search_complete" if root.exists() else "search_root_absent"}
        if root.exists():
            for d,dirs,files in os.walk(root):
                dirs[:]=[x for x in dirs if x not in PRUNE and not x.startswith(".nfs")]
                for fn in files:
                    p=Path(d)/fn; ps=str(p)
                    if ps in seen:continue
                    seen.add(ps);stats["files_considered"]+=1
                    names=[c for c in campaign_names if c in ps]
                    if not names or not safe_text(p):continue
                    txt=text_read(p)
                    if txt is None:stats["errors"]+=1;continue
                    stats["text_files_opened"]+=1
                    matched=[r for r in campaigns if r["campaign"] in ps or r["campaign"] in txt]
                    if len(matched)!=1:continue
                    r=matched[0];k=key(r);stats["campaign_matched_files"]+=1
                    parsed=parse_tag(ps+"\n"+txt[:20000],k[1]); digest=sha_file(p)
                    local_files.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"source_path":ps,"source_sha256":digest,"source_size_bytes":p.stat().st_size,**parsed,"status":"local_metadata_candidate"})
                    if p.suffix.lower()==".json":
                        try:obj=json.loads(txt)
                        except json.JSONDecodeError:obj=None
                        if obj is not None:
                            stats["json_files_parsed"]+=1
                            for field,value in flatten(obj):
                                if KEY_RE.search(field):
                                    structured.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"source_scope":"approved_local_text","source_path":ps,"source_tag":parsed["source_tag"],"field":field,"value":value,"status":"structured_metadata_candidate"})
                    for n,line in enumerate(txt.splitlines(),1):
                        if CFG_RE.search(line):
                            config.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"source_scope":"approved_local_text","source_path":ps,"source_line":n,"line_text":" ".join(line.split())[:1000],"status":"configuration_evidence_candidate"})
        roots.append(stats)

    files_by_tag=defaultdict(list); struct_by_tag=defaultdict(list); cfg_by_campaign=defaultdict(list)
    for r in rep_files+local_files:
        if r["source_tag"]:files_by_tag[r["source_tag"]].append(r)
    for r in structured:
        if r["source_tag"]:struct_by_tag[r["source_tag"]].append(r)
    for r in config:cfg_by_campaign[key(r)].append(r)
    for r in bundle_rows:
        r["exact_metadata_files"]=len(files_by_tag[r["source_tag"]]) if r["source_tag"] else 0
        r["structured_metadata_rows"]=len(struct_by_tag[r["source_tag"]]) if r["source_tag"] else 0
        r["status"]="exact_metadata_matched" if r["structured_metadata_rows"] else "exact_metadata_missing"

    # Duplicate source identities and QCD bin closure.
    duplicates=[]; qcd_groups=defaultdict(list)
    for k,rows in by_campaign.items():
        ids=defaultdict(list)
        for r in rows:ids[(r["bin_index"],r["shard_id"],r["seed"])].append(r["bundle_basename"])
        for ident,names in ids.items():
            if len(names)>1:duplicates.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"bin_index":ident[0],"shard_id":ident[1],"seed":ident[2],"bundles":",".join(names),"status":"duplicate_identity"})
        if k[1]=="qcd_hardqcd":
            for r in rows:
                if r["bin_index"]=="":raise RuntimeError(f"unparsed QCD identity {r['bundle_basename']}")
                qcd_groups[(k[2],r["bin_index"],r["pthat_min_GeV"],r["pthat_max_GeV"])].append(r)
    qcd_bins=[]
    for (camp,idx,lo,hi),rows in sorted(qcd_groups.items(),key=qcd_group_sort_key):
        covered=sum(bool(r["structured_metadata_rows"]) for r in rows)
        qcd_bins.append({"campaign":camp,"bin_index":idx,"pthat_min_GeV":lo,"pthat_max_GeV":hi,"bundle_count":len(rows),"bundles_with_structured_metadata":covered,"identities_unique":len({r["source_tag"] for r in rows})==len(rows),"interval_valid":pthat_interval_valid(lo,hi),"metadata_coverage_complete":covered==len(rows),"qcd_stitching_authorized":False,"status":"complete_candidate" if covered==len(rows) else "recovery_incomplete"})

    status=[]; plan=[]
    for r in registry:
        k=key(r);rows=by_campaign[k];covered=sum(bool(x["structured_metadata_rows"]) for x in rows);cfg=len(cfg_by_campaign[k])
        denominator_complete=covered==len(rows)
        config_complete=cfg>0
        status.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"expected_bundles":int(r["members"]),"eos_bundles_listed":len(rows),"expected_events":int(r["generated_events"]),"bundles_with_structured_metadata":covered,"configuration_evidence_rows":cfg,"denominator_metadata_coverage_complete":denominator_complete,"configuration_evidence_present":config_complete,"normalization_denominator_authorized":False,"qcd_stitching_authorized":False,"physical_weight_authorized":False,"required_next_action":"freeze nonstandard conventions" if denominator_complete and config_complete else "recover missing exact shard metadata/configuration","status":"reviewed_fail_closed"})
        for x in rows:
            if x["structured_metadata_rows"]:continue
            plan.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"remote_bundle_path":x["remote_bundle_path"],"source_tag":x["source_tag"],"missing_artifact":"per_shard_structured_generator_metadata","preferred_recovery":"local_condor_receipt_before_bundle_transfer","payload_access_authorized":False,"status":"targeted_recovery_required"})
        if not config_complete:
            plan.append({"sample_class":k[0],"process_or_mode":k[1],"campaign":k[2],"remote_bundle_path":"","source_tag":"","missing_artifact":"exact_generator_configuration_and_decay_scope","preferred_recovery":"generation_script_card_compile_arguments_or_receipt","payload_access_authorized":False,"status":"targeted_recovery_required"})

    write_tsv(a.output/"nonstandard_eos_bundle_ledger.tsv",bundle_rows,list(bundle_rows[0]))
    write_tsv(a.output/"representative_provenance_source_files.tsv",rep_files,list(rep_files[0]))
    write_tsv(a.output/"local_nonstandard_metadata_files.tsv",local_files,list(local_files[0]) if local_files else ["sample_class","process_or_mode","campaign","source_path","source_sha256","source_size_bytes","source_tag","bin_index","pthat_min_GeV","pthat_max_GeV","shard_id","seed","status"])
    write_tsv(a.output/"structured_nonstandard_metadata_candidates.tsv",structured,list(structured[0]))
    write_tsv(a.output/"nonstandard_configuration_evidence.tsv",config,list(config[0]))
    write_tsv(a.output/"qcd_bin_metadata_closure.tsv",qcd_bins,list(qcd_bins[0]))
    write_tsv(a.output/"duplicate_nonstandard_source_identities.tsv",duplicates,list(duplicates[0]) if duplicates else ["sample_class","process_or_mode","campaign","bin_index","shard_id","seed","bundles","status"])
    write_tsv(a.output/"nonstandard_campaign_metadata_status.tsv",status,list(status[0]))
    write_tsv(a.output/"targeted_nonstandard_recovery_plan.tsv",plan,list(plan[0]) if plan else ["sample_class","process_or_mode","campaign","remote_bundle_path","source_tag","missing_artifact","preferred_recovery","payload_access_authorized","status"])
    write_tsv(a.output/"approved_search_root_summary.tsv",roots,list(roots[0]))

    complete=sum(bool(r["denominator_metadata_coverage_complete"]) for r in status)
    cfgcomplete=sum(bool(r["configuration_evidence_present"]) for r in status)
    binscomplete=sum(bool(r["metadata_coverage_complete"]) for r in qcd_bins)
    next_gate="freeze_nonstandard_denominators_configuration_and_qcd_stitching" if complete==6 and cfgcomplete==6 and binscomplete==len(qcd_bins) and not duplicates else "execute_targeted_nonstandard_metadata_recovery_plan"
    summary={
      "schema_version":1,"status":"hh4b_nonstandard_source_metadata_closure_pass",
      "timestamp_utc":datetime.now(timezone.utc).isoformat(),"source_commit":a.source_commit,
      "inventory":{"nonstandard_campaigns":6,"ggf_campaigns":2,"qcd_hardqcd_campaigns":4,"eos_bundles_listed":len(bundle_rows),"expected_eos_bundles":expected_bundles,"expected_generated_events":expected_events,"qcd_bin_groups":len(qcd_bins)},
      "metadata":{"representative_provenance_files_reverified":len(rep_files),"local_campaign_matched_text_files":len(local_files),"structured_metadata_candidate_rows":len(structured),"configuration_evidence_rows":len(config),"duplicate_source_identities":len(duplicates),"campaigns_with_complete_denominator_metadata":complete,"campaigns_with_configuration_evidence":cfgcomplete,"qcd_bins_with_complete_denominator_metadata":binscomplete,"targeted_recovery_rows":len(plan)},
      "readiness":{"nonstandard_source_ledger_complete":True,"nonstandard_denominators_authorized":0,"qcd_stitching_authorized":False,"external_reference_cross_sections_authorized":0,"physical_weight_application_authorized":False,"physics_normalization_ready":False},
      "controls":{"root_files_opened":0,"hepmc_files_opened":0,"lhe_files_opened":0,"candidate_files_opened":0,"candidate_rows_read":0,"validation_candidate_files_opened":0,"evaluation_candidate_files_opened":0,"nonstandard_denominators_authorized":0,"qcd_stitching_rules_authorized":0,"cross_sections_assigned":0,"physical_weights_calculated":0,"physical_yields_calculated":0,"models_trained":0,"thresholds_selected":0},
      "next_gate":next_gate}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (a.output/"README.md").write_text("# HH4b nonstandard source-metadata closure\n\nThis checkpoint freezes all 361 EOS bundle identities for the two ggF HH and four importance-sampled hard-QCD campaigns, rechecks representative provenance, and searches approved local text-only production roots for exact shard metadata and generator configuration evidence. No ROOT, HepMC, LHE, Parquet, candidate, validation, or evaluation payload is opened. No denominator, stitching rule, cross section, physical weight, or yield is authorized.\n")
    products=sorted(p for p in a.output.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (a.output/"SHA256SUMS").write_text("\n".join(f"{sha_file(p)}  {p.relative_to(a.output)}" for p in products)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))
    print("ALL_SIX_NONSTANDARD_CAMPAIGNS_SOURCE_LEDGERED_PASS")
    print("ALL_361_NONSTANDARD_EOS_BUNDLES_LISTED_PASS")
    print("QCD_PTHAT_BIN_IDENTITIES_PARSED_PASS")
    print("TARGETED_NONSTANDARD_RECOVERY_PLAN_FROZEN_PASS")
    print("NO_ROOT_HEPMC_LHE_OR_CANDIDATE_PAYLOAD_OPENED")
    print("NO_NONSTANDARD_DENOMINATOR_AUTHORIZED")
    print("NO_QCD_STITCHING_RULE_AUTHORIZED")
    print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
    print("NO_PHYSICAL_WEIGHTS_OR_YIELDS_CALCULATED")
    print("HH4B_NONSTANDARD_SOURCE_METADATA_CLOSURE_PASS")
if __name__=="__main__":main()
