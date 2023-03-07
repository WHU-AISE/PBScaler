package contacts.repository;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import contacts.entity.Contacts;
import java.util.ArrayList;
import java.util.UUID;

/**
 * @author fdse
 */
@Repository
public interface ContactsRepository extends MongoRepository<Contacts, String> {

    /**
     * find by id
     *
     * @param id id
     * @return Contacts
     */
    Contacts findById(UUID id);

    /**
     * find by account id
     *
     * @param accountId account id
     * @return ArrayList<Contacts>
     */
    @Query("{ 'accountId' : ?0 }")
    ArrayList<Contacts> findByAccountId(UUID accountId);

    /**
     * delete by id
     *
     * @param id id
     * @return null
     */
    void deleteById(UUID id);

    /**
     * find all
     *
     * @return ArrayList<Contacts>
     */
    @Override
    ArrayList<Contacts> findAll();

}
