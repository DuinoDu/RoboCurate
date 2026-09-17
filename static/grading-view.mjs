// Legacy ratings used 100 points; version 3 exposes max_score=10.
export function ratingPoints(rating, value=rating?.score) {
  if(!Number.isFinite(value))return null;
  const maximum=rating?.max_score??100;
  return Number.isFinite(maximum)&&maximum>0?value*10/maximum:null;
}
export const ratingText=(rating,value=rating?.score)=>ratingPoints(rating,value)?.toFixed(2)??'—';
export const ratingPercent=(rating,value=rating?.score)=>Math.max(0,Math.min(100,(ratingPoints(rating,value)??0)*10));

// Pending analysis is not a poor grade and always stays after measured data.
export function compareDataGrades(a, b, bestFirst = true) {
  const x = a.quality.grading, y = b.quality.grading;
  if (!x?.grade || !y?.grade) return Number(!x?.grade) - Number(!y?.grade);
  const direction = bestFirst ? 1 : -1;
  return direction * (x.grade.localeCompare(y.grade) ||
    (ratingPoints(b.quality.rating) ?? -1) - (ratingPoints(a.quality.rating) ?? -1) ||
    (y.metrics.coverage_fraction ?? -1) - (x.metrics.coverage_fraction ?? -1)) ||
    (a.episode_id || a.name).localeCompare(b.episode_id || b.name);
}
