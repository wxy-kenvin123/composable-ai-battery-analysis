#!/usr/bin/env python3
"""Offline static scoring of newly acquired, paper-attributed source snapshots.

Never clones, downloads, executes repository code, or applies historical overrides.
The copied scoring_source.py is imported only for its read-only scanners and
boolean rules. Positives have file/cell evidence; negatives identify scan scope.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, pathlib, re, sys
sys.dont_write_bytecode = True
import scoring_source as src

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDICATORS = ['R1','R2','R3','R4','R5','R6','R7','D1','D2','D3']
CODE_EXTS = {'.py','.ipynb','.m','.r','.jl','.sh','.bash','.c','.cpp','.cc','.cxx','.h','.hpp','.f','.f90','.f95','.for','.java','.js','.ts','.rs','.go','.pl','.rb','.tcl','.vi'}

def truth(value):
    return str(value).strip().lower() in {'true','1','yes','y'}

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def normalize_manifest(row):
    r=dict(row)
    for key,alternatives in {
        'paper_id':['id'],'resource_id':['normalized_url','canonical_url','url'],
        'snapshot_path':['local_path','repo_path','cache_path'],
        'source_url':['url'],'final_url':['canonical_url','url'],
        'acquisition_status':['fetch_status','status'],'revision':['commit','commit_sha'],
        'snapshot_sha256':['sha256'],'resource_kind':['kind'],'attribution':['ownership'],
        'is_primary':['primary_code','primary'],
    }.items():
        if not r.get(key):
            r[key]=next((row[a] for a in alternatives if row.get(a)), '')
    return r

def snapshot_path(row):
    if not str(row.get('snapshot_path','')).strip():raise ValueError('Missing resource snapshot path; refusing run-root scan')
    p=pathlib.Path(row['snapshot_path'])
    if not p.is_absolute(): p=ROOT/p
    p=p.resolve()
    if p==ROOT.resolve() or not p.is_relative_to(ROOT): raise ValueError('Snapshot is outside this run: '+str(p))
    return p

def manifest_eligible(row):
    return (bool(str(row.get('snapshot_path','')).strip())
            and bool(row.get('code_evidence_file'))
            and str(row.get('acquisition_status','')).lower() in {'verified','downloaded','cloned','success','ok'}
            and str(row.get('attribution','')).lower() in {'paper_specific','author_resource','author','paper-specific'}
            and str(row.get('resource_kind','')).lower() in {'code','both','code/data','code_and_data'})

def score(scan):
    f=scan['files']; rd=scan['readme']; dp=scan['dep_text']; py=scan['py_text']; nb=scan['notebook_text']; mt=scan['matlab_text']
    r={'R1':True,'R2':src.check_r2(f,rd),'R3':src.check_r3(f,rd),'R4':src.check_r4(f),'R5':src.check_r5(f,rd),'R6':src.check_r6(f,rd,scan['tree_data']),'R7':src.check_r7(f)}
    d={'D1':src.check_d1(f,dp,py,rd,nb,mt),'D2':src.check_d2(dp,py,rd,nb,scan['mson_text'],scan['chem_files'],scan['mson_files'],scan['table_d2_headers']),'D3':src.check_d3(f,py,dp,scan['executable_notebooks'],scan['matlab_main_scripts'])}
    return {**r,**d,'RRL':src.compute_rrl(r),'CRL':src.compute_crl(d)}

def content_units(repo,scan):
    """Read only text included by the original capped scanner, preserving cells."""
    units=[]
    for kind in ['dep_text','py_text','notebook_text','matlab_text','mson_text']:
        text=scan[kind]
        parts=re.split(r'\n[#%] --- (.*?) ---\n',text)
        for i in range(1,len(parts),2):
            rel,body=parts[i],parts[i+1]
            if kind=='notebook_text':
                try:
                    data=json.loads((repo/rel).read_text(encoding='utf-8',errors='replace'))
                    for cell_i,cell in enumerate(data.get('cells',[]),1):
                        if cell.get('cell_type')!='code': continue
                        src=cell.get('source',''); src=''.join(src) if isinstance(src,list) else str(src)
                        units.append({'file':rel,'kind':kind,'text':src,'cell':cell_i})
                except (OSError,ValueError): pass
            else: units.append({'file':rel,'kind':kind,'text':body,'cell':''})
    for rel in scan['files']:
        if pathlib.PurePosixPath(rel).name.lower().startswith('readme'):
            try:
                body=(repo/rel).read_text(encoding='utf-8',errors='replace')
                if body==scan['readme'] and body:
                    units.append({'file':rel,'kind':'readme','text':body,'cell':''}); break
            except OSError: pass
    return units

def is_pointer(path):
    with open(path,'rb') as f:head=f.read(256).lstrip()
    return head.startswith(b'version https://git-lfs.github.com/spec/v1')

def evidence_for(repo,scan,ind,code_files,review_reason=""):
    """Return minimal positive witnesses without changing original rules."""
    units=content_units(repo,scan)
    def file_ev(rel,note='file-presence rule'):
        p=repo/rel; suffix=p.suffix.lower(); binary=suffix in src.BINARY_EXTS or suffix in {'.xlsx','.vi'}
        loc=f'{rel} [file metadata]' if binary else f'{rel}:1'
        excerpt=note
        if suffix=='.ipynb':
            n=next((u for u in units if u['file']==rel and u['text'].strip()),None)
            if n:
                first=next((i for i,line in enumerate(n['text'].splitlines(),1) if line.strip()),1)
                loc=f'{rel}#cell-{n["cell"]}:{first}'; excerpt=n['text'].splitlines()[first-1][:300]
        return {'file':rel,'locator':loc,'excerpt':excerpt,'rule_detail':note}
    def search(pattern,kinds=None,flags=0,only_file=None):
        for u in units:
            if kinds and u['kind'] not in kinds: continue
            if only_file is not None and u['file']!=only_file:continue
            m=re.search(pattern,u['text'],flags)
            if m:
                n=u['text'][:m.start()].count('\n')+1
                loc=f'{u["file"]}#cell-{u["cell"]}:{n}' if u['cell'] else f'{u["file"]}:{n}'
                lines=u['text'].splitlines(); excerpt='\n'.join(lines[max(0,n-1):min(len(lines),n+1)])[:500]
                return {'file':u['file'],'locator':loc,'excerpt':excerpt,'rule_detail':pattern}
        return None
    f=scan['files']; rd=scan['readme']; rd_kind={'readme'}
    if ind=='R1':
        e=file_ev(code_files[0],'Current-run content-reviewed source payload; original hash matches snapshot')
        if review_reason:e['locator']+=' | '+review_reason;e['excerpt']=review_reason
        return [e]
    if ind=='R2':
        for rel in f:
            if src.check_r2([rel],''): return [file_ev(rel)]
        return [search(r'(?:pip|pip3)\s+install\s+(?!-e\s+\.)(?!\.)[a-zA-Z][\w\-]+|conda\s+install\s+(?:-[cy]\s+\S+\s+)?[a-zA-Z][\w\-]+',rd_kind)]
    if ind=='R3':
        for rel in f:
            if rel.endswith('.ipynb'): return [file_ev(rel,'static counts notebook file presence as run instructions')]
        return [search(r'```(bash|sh|shell|console|python)',rd_kind,re.I) or search(r'(pip install|python |conda |pip |bash |sh )',rd_kind)]
    if ind=='R4': return [file_ev(next(rel for rel in f if src.check_r4([rel])))]
    if ind=='R5':
        for rel in f:
            if src.check_r5([rel],'') and not is_pointer(repo/rel): return [file_ev(rel)]
        hit=search(r'(pretrained|checkpoint|model.*weight|download.*model|\.pt\b|\.pth\b|\.ckpt\b)',rd_kind,re.I)
        if hit:return [hit]
        return [file_ev(rel,'Raw static filename trigger is a Git LFS pointer, not acquired payload; final item requires separate evidence review') for rel in f if src.check_r5([rel],'')][:1]
    if ind=='R6':
        for rel in f:
            if src.check_r6([rel],'',scan['tree_data']) and not is_pointer(repo/rel): return [file_ev(rel,f'static data-file rule; size={(repo/rel).stat().st_size} bytes')]
        hit=search(r'(zenodo\.org|figshare\.com|huggingface\.co|download.*data|data.*available.*http)',rd_kind,re.I)
        if hit:return [hit]
        return [file_ev(rel,'Raw static file trigger is a Git LFS pointer, not acquired payload; final item requires separate evidence review') for rel in f if src.check_r6([rel],'',scan['tree_data'])][:1]
    if ind=='R7': return [file_ev(next(rel for rel in f if src.check_r7([rel])))]
    code={'py_text','notebook_text','matlab_text'}
    if ind=='D1':
        d1_kinds={'dep_text','py_text','notebook_text','matlab_text','readme'}
        hit=search(r'deepmd|nequip|aiida|mace-torch|mace_torch|schnet|e3nn|allegro|dp_train|dpdata|chgnet|m3gnet|mp-api|mp_api|matgl',d1_kinds,re.I)
        if hit: return [hit]
        a=search('lammps',d1_kinds,re.I); b=search(r'pair_style|pair_coeff|in\.lammps',d1_kinds,re.I)
        if a and b: return [a,b]
        cf=[rel for rel in f if any(t in rel.lower() for t in ['config','cfg','params']) and pathlib.PurePosixPath(rel).suffix in {'.yaml','.yml','.json','.toml','.cfg','.ini'}]
        hit=search(r'yaml\.(?:safe_)?load|json\.load\(|(?:load|read|parse)_config|config.*=.*(?:yaml|json)\.(?:safe_)?load|OmegaConf|hydra|configparser',code)
        if hit and cf:return [hit,file_ev(cf[0],'configuration file required alongside configuration loader')]
        a=search(r'argparse\.ArgumentParser|parser\.add_argument',code);b=search(r'--(?:config|input|data|dataset|structure|material|params?)\b',code)
        if a and b:return [a,b]
        for kind in ['notebook_text','matlab_text']:
            if src._code_has_external_input_loader(scan[kind]):
                pats=[r'\bload\s+[A-Za-z0-9_./\\ -]+\.(?:mat|csv|tsv|xlsx|xls|txt|dat)\b',r'(?:pd\.)?read_(?:csv|excel|table)\s*\(|np\.load\s*\(|loadmat\s*\(|scipy\.io\.loadmat\s*\(|readtable\s*\(|readmatrix\s*\(|xlsread\s*\(|csvread\s*\(|importdata\s*\(|\bload\s*\(',r'[\'\"][^\'\"]+\.(?:csv|tsv|xlsx|xls|txt|json|npy|npz|mat|dat)[\'\"]',r'\b(?:data|input|file|filename|filepath|path|datadir|data_dir)\w*\s*=\s*']
                pats.append(r'(?:read_(?:csv|excel|table)|np\.load|loadmat|readtable|readmatrix|xlsread|csvread|importdata|\bload)\s*\(\s*(?:data|input|file|filename|filepath|path|datadir|data_dir)\w*')
                return [x for pat in pats if (x:=search(pat,{kind},re.I))]
    if ind=='D2':
        kinds={'dep_text','py_text','notebook_text','mson_text'}
        hit=search(r'pymatgen|rdkit|openbabel|mdanalysis|matminer|(?:import\s+ase|from\s+ase[\s.]|ase\.io|ase\.atoms|ase\.build|\base[=><])',kinds,re.I)
        if hit:return [hit]
        if scan['chem_files']: return [file_ev(scan['chem_files'][0],'standard chemistry file rule')]
        if scan['mson_files']:return [file_ev(scan['mson_files'][0],'pymatgen/ASE MSON metadata')]
        if scan['table_d2_headers']:
            rel,matched=scan['table_d2_headers'][0].split(': ',1)
            ev=file_ev(rel,'table header matches: '+matched)
            header=src._read_table_header(repo/rel,rel)
            ev['excerpt']=' | '.join(header)[:500]
            if pathlib.PurePosixPath(rel).suffix.lower() in {'.csv','.tsv'}:
                with open(repo/rel,encoding='utf-8-sig',errors='replace') as f:
                    for line_no,line in enumerate(f,1):
                        if line.strip():ev['locator']=f'{rel}:{line_no} (header)';break
            else:
                with src.zipfile.ZipFile(repo/rel) as zf:
                    name=sorted(n for n in zf.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$',n))[0]
                    root=src.ET.fromstring(zf.read(name));ns='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
                    for r in root.findall(f'.//{ns}sheetData/{ns}row'):
                        if any((c.find(f'{ns}v') is not None or c.find(f'{ns}is') is not None) for c in r.findall(f'{ns}c')):
                            ev['locator']=f'{rel}#{name}:row-{r.attrib.get("r","?")}';break
            return [ev]
    if ind=='D3':
        hit=search(r'argparse\.ArgumentParser|parser\.add_argument|click\.command|fire\.Fire|typer\.Typer|click\.group|click\.option',{'py_text'})
        if hit:return [hit]
        hit=search('console_scripts',{'dep_text'})
        if hit:return [hit]
        install=[rel for rel in f if pathlib.PurePosixPath(rel).name in {'setup.py','setup.cfg','pyproject.toml'}]
        init=[rel for rel in f if rel.endswith('__init__.py')]
        if install and init:return [file_ev(install[0],'installable package metadata'),file_ev(init[0],'package initializer')]
        if scan['executable_notebooks']:return [file_ev(scan['executable_notebooks'][0],'top/first-level nonempty code notebook')]
        if scan['matlab_main_scripts']:
            rel=scan['matlab_main_scripts'][0]
            ev=search(r'\breadtable\s*\(|\breadmatrix\s*\(|\bxlsread\s*\(|\bcsvread\s*\(|\bload\s*\(|\bimportdata\s*\(|\bfitr?\w*\s*\(|\btrain\w*\s*\(|\bpredict\s*\(|\bsave\s*\(|\bwritetable\s*\(|\bplot\s*\(|=|\bfor\b|\bif\b',{'matlab_text'},re.I,rel)
            return [ev or file_ev(rel,'static recognized top/first-level MATLAB main script')]
    return []

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=pathlib.Path,default=ROOT/'data/resource_manifest.csv');args=ap.parse_args()
    rows=[normalize_manifest(r) for r in read_csv(args.manifest)]
    selected=[r for r in rows if truth(r.get('is_primary')) and r.get('resource_kind','').lower() in {'code','both','code/data','code_and_data'}]
    seen=set()
    for r in selected:
        if r['paper_id'] in seen:raise ValueError('Multiple primary resources for paper '+r['paper_id'])
        seen.add(r['paper_id'])
    papers=[]; resources=[]; evidence=[];inventory=[];cache={}; errors=[]
    for row in selected:
        pid=row['paper_id'];rid=row['resource_id'];key=(rid,row['snapshot_path'])
        meta={k:row.get(k,'') for k in ['paper_id','resource_id','source_url','final_url','snapshot_path','retrieved_utc','revision','snapshot_sha256','acquisition_status','attribution']}
        if not manifest_eligible(row):
            papers.append({**meta,'scoring_status':'not_scored_unverified_or_unattributed','R1':'',**{k:'' for k in INDICATORS[1:]},'RRL':'','CRL':'','note':'Unavailable/unverified is not coded as zero.'});continue
        try:
            repo=snapshot_path(row)
            if not repo.is_dir():raise ValueError('Snapshot directory missing: '+str(repo))
            if key not in cache:
                scan=src.scan_local_repo(repo)
                code_files=[]
                for rel in scan['files']:
                    p=repo/rel
                    if p.is_symlink():raise ValueError('Symlink forbidden in scoring snapshot: '+rel)
                    size=p.stat().st_size
                    is_code=p.suffix.lower() in CODE_EXTS and size>0 and p.name not in {'__init__.py','setup.py'}
                    if is_code:
                        with p.open('rb') as source_stream:head=source_stream.read(1024).strip()
                        is_code=bool(head) and not is_pointer(p) and not head.lower().startswith((b'<!doctype html',b'<html'))
                    if is_code and p.suffix.lower()=='.ipynb':is_code=bool(src._extract_notebook_code(p).strip())
                    if is_code:code_files.append(rel)
                    inventory.append({'resource_id':rid,'snapshot_path':row['snapshot_path'],'relative_path':rel,'bytes':size,'sha256':digest(p),'computational_source_candidate':is_code})
                # static R1 concerns accessible code, not the extension. The
                # content review of the current run may identify native VI, plain
                # TXT, extensionless code, or embedded source in an SI PDF.
                witness=row.get('code_evidence_file','')
                if witness:
                    original=pathlib.Path(witness)
                    if not original.is_absolute():original=ROOT/original
                    expected=row.get('code_evidence_sha256','')
                    if not original.is_file() or digest(original)!=expected:
                        raise ValueError('Reviewed code witness hash mismatch')
                    choices=[rel for rel in scan['files'] if (repo/rel).stat().st_size==original.stat().st_size]
                    matched=next((rel for rel in choices if digest(repo/rel)==expected),None)
                    if not matched:raise ValueError('Reviewed source payload is not in the chosen snapshot; refusing filename fallback')
                    code_files=[matched]+[f for f in code_files if f!=matched]
                if not code_files:
                    cache[key]=({'R1':'',**{k:'' for k in INDICATORS[1:]},'RRL':'','CRL':'','scoring_status':'not_scored_no_computational_payload'},[])
                else:
                    values=score(scan); evs=[]
                    for ind in INDICATORS:
                        if values[ind]:
                            witnesses=[e for e in evidence_for(repo,scan,ind,code_files,row.get('code_evidence_reason','')) if e]
                            if not witnesses:raise ValueError('Positive rule has no localized witness: '+ind)
                        else:
                            witnesses=[{'file':'','locator':'data/scoring_file_inventory.csv [resource_id='+rid+']','excerpt':'No match in the static bounded static scan. This is not proof that the feature does not exist.','rule_detail':'Original static checker; scan caps and excluded paths unchanged.'}]
                        for e in witnesses:evs.append({'indicator':ind,'value':values[ind],**e})
                    values['scoring_status']='scored';cache[key]=(values,evs)
                resources.append({'resource_id':rid,'snapshot_path':row['snapshot_path'],'file_count':len(scan['files']),'code_files':len(code_files),'source_rules_sha256':digest(pathlib.Path(src.__file__)),'scoring_status':cache[key][0]['scoring_status']})
            values,evs=cache[key]
            papers.append({**meta,**values,'note':'Static static; no scientific program executed; historical overrides disabled.'})
            for e in evs:evidence.append({'paper_id':pid,'resource_id':rid,'snapshot_path':row['snapshot_path'],'item':e['indicator'],**e})
        except Exception as exc:
            errors.append({'paper_id':pid,'resource_id':rid,'error':str(exc)})
            papers.append({**meta,'scoring_status':'scan_failed','R1':'',**{k:'' for k in INDICATORS[1:]},'RRL':'','CRL':'','note':str(exc)})
    write_csv(ROOT/'data/scoring_papers_raw.csv',papers,['paper_id','resource_id','source_url','final_url','snapshot_path','retrieved_utc','revision','snapshot_sha256','acquisition_status','attribution','scoring_status',*INDICATORS,'RRL','CRL','note'])
    write_csv(ROOT/'data/scoring_resources_raw.csv',resources,['resource_id','snapshot_path','file_count','code_files','source_rules_sha256','scoring_status'])
    write_csv(ROOT/'data/scoring_evidence_raw.csv',evidence,['paper_id','resource_id','snapshot_path','item','indicator','value','file','locator','excerpt','rule_detail'])
    write_csv(ROOT/'data/scoring_file_inventory.csv',inventory,['resource_id','snapshot_path','relative_path','bytes','sha256','computational_source_candidate'])
    write_csv(ROOT/'data/scoring_errors.csv',errors,['paper_id','resource_id','error'])
    summary={'primary_papers':len(papers),'scored_papers':sum(r.get('scoring_status')=='scored' for r in papers),'unique_snapshots':len(resources),'evidence_rows':len(evidence),'errors':len(errors),'historical_overrides_applied':0,'execution':'static only','source_rules_sha256':digest(pathlib.Path(src.__file__))}
    (ROOT/'data/scoring_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));return 1 if errors else 0

if __name__=='__main__':raise SystemExit(main())
