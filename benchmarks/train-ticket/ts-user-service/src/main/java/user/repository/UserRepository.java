package user.repository;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;
import user.entity.User;

import java.util.UUID;

/**
 * @author fdse
 */
@Repository
public interface UserRepository extends MongoRepository<User, String> {

    User findByUserName(String userName);

    User findByUserId(UUID userId);

    void deleteByUserId(UUID userId);
}
