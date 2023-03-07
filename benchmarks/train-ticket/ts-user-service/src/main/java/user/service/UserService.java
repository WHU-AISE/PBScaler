package user.service;

import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;
import user.dto.UserDto;

import java.util.UUID;

/**
 * @author fdse
 */
public interface UserService {
    Response saveUser(UserDto user, HttpHeaders headers);

    Response getAllUsers(HttpHeaders headers);

    Response findByUserName(String userName, HttpHeaders headers);
    Response findByUserId(String userId, HttpHeaders headers);


    Response deleteUser(UUID userId, HttpHeaders headers);

    Response updateUser(UserDto user, HttpHeaders headers);
}
