// compact sidebar toggle
document.addEventListener('DOMContentLoaded', function(){
  var btn = document.getElementById('sidebarToggle');
  var container = document.querySelector('.container');
  var sidebar = document.querySelector('.sidebar');
  if(!btn || !container) return;
  btn.onclick = function(){
    var collapsed = container.classList.toggle('sidebar-collapsed');
    btn.setAttribute('aria-expanded', collapsed ? 'true' : 'false');
    if(sidebar) sidebar.style.display = collapsed ? 'none' : '';
  };
  window.addEventListener('resize', function(){
    if(window.innerWidth>900){ container.classList.remove('sidebar-collapsed'); if(sidebar) sidebar.style.display=''; btn.setAttribute('aria-expanded','false'); }
  });
});
