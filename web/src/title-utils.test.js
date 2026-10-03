import test from 'node:test';
import assert from 'node:assert/strict';
import { applyTitle, createTitlePrompt, describeMelody, extractGeneratedTitle, fallbackTitle, isRepeatedTitle, sanitizeTitle } from './title-utils.js';

test('naming preserves every musical byte and event and keeps RTTTL names unique', () => {
  const song = { id:'batch-1',name:'Sketch',rtttl:'Sketch:d=8,o=5,b=90:c,d,e',bpm:90,events:[{pitch:72,duration:8,dotted:false}] };
  const titles=new Set(),names=new Set();
  const first=applyTitle(song,'"Silver Morning"',titles,names);
  assert.throws(()=>applyTitle(song,'Silver Morning',titles,names), /repeated/);
  const second=applyTitle(song,'Silver Morninglight',titles,names);
  assert.equal(first.rtttl.split(':').slice(1).join(':'),song.rtttl.split(':').slice(1).join(':'));
  assert.equal(first.events,song.events);
  assert.equal(first.id,song.id);
  assert.notEqual(first.name,second.name);
  assert.notEqual(first.title,second.title);
  assert.ok(second.name.length<=11);
  assert.equal(song.name,'Sketch');
  assert.equal(sanitizeTitle('"Silver Morning".'), 'Silver Morning');
});

test('title prompt uses measured music without embedding source names or RTTTL text', () => {
  const description=describeMelody({name:'PRIVATE SOURCE',bpm:75,events:[{pitch:72,duration:2,dotted:false}]});
  assert.match(description,/slow melody at 75 BPM/);
  assert.doesNotMatch(description,/PRIVATE/);
  assert.match(description,/sustained notes/);
  const fast = describeMelody({bpm:170,settings:{profile:'chiptune'},events:[{pitch:96,duration:16},{pitch:84,duration:16},{pitch:96,duration:16}]});
  assert.match(fast,/wide melodic jumps/);
  assert.match(fast,/playful game music inspiration/);
  assert.notEqual(description,fast);
});

test('duplicate and near-duplicate model titles trigger another suggestion instead of number suffixes', () => {
  const used = new Set(['Silver Morning', 'Lamps in the Rain', 'Deep Blue Sky']);
  for (const title of ['silver morning', 'Silver Morning 2', 'The Silver Morning', 'Rain Lamps', 'Blue Sky Soundtrack']) {
    assert.equal(isRepeatedTitle(title,used),true,title);
  }
  assert.equal(isRepeatedTitle('Copper Sunset',used),false);
  const prompt=createTitlePrompt({bpm:80,events:[]},used);
  assert.match(prompt,/Silver Morning; Lamps in the Rain/);
  assert.doesNotMatch(prompt,/Pixel Fireflies|Velvet Moon|Sunlit Steps/);
  assert.notEqual(prompt,createTitlePrompt({bpm:80,events:[]},used,1));
});

test('unusable or repeated LLM names have distinct grounded fallbacks with every tune preserved', () => {
  const song={id:'preserved-id',name:'Original',bpm:165,events:[{pitch:90,duration:8}],rtttl:'Original:d=8,o=6,b=165:f#'};
  const titles=new Set(),names=new Set();
  const results=Array.from({length:24},()=>fallbackTitle(song,titles,names));
  assert.equal(new Set(results.map(result=>result.title)).size,24);
  assert.equal(new Set(results.map(result=>result.name)).size,24);
  for(const result of results) {
    assert.equal(result.id,song.id);
    assert.equal(result.events,song.events);
    assert.equal(result.rtttl.slice(result.rtttl.indexOf(':')),song.rtttl.slice(song.rtttl.indexOf(':')));
    assert.equal(result.naming.status,'fallback');
    assert.equal(result.naming.strategy,'musical-character');
    assert.doesNotMatch(result.title,/\d/);
  }
});

test('completion parsing isolates one title and removes trailing performance notes', () => {
  assert.equal(extractGeneratedTitle(' Dancing Boots\nSound: Another tune.\nTitle: Wrong Tune'), 'Dancing Boots');
  assert.equal(extractGeneratedTitle('"Wingsuit Dreams" (Voice Only)'), 'Wingsuit Dreams');
  assert.throws(()=>extractGeneratedTitle('<script>Bad</script>\nNice Title'));
  assert.throws(()=>extractGeneratedTitle('This melody is beautiful\nNice Title'));
});

test('invalid model responses cannot become markup or extra RTTTL sections', () => {
  for (const text of ['<script>alert(1)</script>','Hello:world','Good\nMorning','Here is a title for this song','1234','SingsandshellSonglightssinging']) {
    assert.throws(()=>sanitizeTitle(text));
  }
});
