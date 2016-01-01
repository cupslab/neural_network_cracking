var factory = function(cb) {
  return {
    query_guess_number : function(pwd) {
      window.setTimeout(function(){
        cb(zxcvbn(pwd, []).guesses, pwd);
      }, 1);
    }
  };
};
