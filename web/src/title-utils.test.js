import test from 'node:test';
import assert from 'node:assert/strict';
import { applyTitle, describeMelody, sanitizeTitle } from './title-utils.js';

test('naming preserves every musical byte and event and keeps RTTTL names unique', () => {
  const song = { name:'Sketch',rtttl:'Sketch:d=8,o=5,b=90:c,d,e',bpm:90,events:[{pitch:72,duration:8,dotted:false}] };
  const titles=new Set(),names=new Set();
  const first=applyTitle(song,'"Silver Morning"',titles,names);
  const second=applyTitle(song,'Silver Morning',titles,names);
  assert.equal(first.rtttl.split(':').slice(1).join(':'),song.rtttl.split(':').slice(1).join(':'));
  assert.equal(first.events,song.events);
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
});

test('invalid model responses cannot become markup or extra RTTTL sections', () => {
  for (const text of ['<script>alert(1)</script>','Hello:world','Good\nMorning','Here is a title for this song','1234','SingsandshellSonglightssinging']) {
    assert.throws(()=>sanitizeTitle(text));
  }
});
